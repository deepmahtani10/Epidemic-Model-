#include <SFML/Graphics.hpp>
#include <Eigen/Dense>
#include <GPs.h>
#include <LBFGS.h>
#include <thread>
#include <vector>
#include <random>
#include <atomic>
#include <mutex>
#include <condition_variable>
#include <chrono>
#include <iostream>

using Eigen::MatrixXd;
using Eigen::VectorXd;
using GP::RBF;
GP::GaussianProcess gp;
std::atomic<bool> ode_ready{false}; // used to help with thread race problems and unreliable outputs, I will try to find a way to synchronzie the threads
std::atomic<bool> gp_ready{false};
std::condition_variable ode_data_condvar;
std::condition_variable gp_data_condvar;
// bool GaussianSkip = false;
bool isStillSampling = false;

int N = 144;
float pInfect = 0.0;
float Beta = 3.0f; // infection rate paramater, as stated from README file, this is a value I adopted without any reason beyond the fact that I like it.

float AlterRetP = 0.0f; // A control variable used in GuassianProcessProbI(t_monte) function call that will help when the threads are running but not all data samples required have been collected, I intend to use this to allow the threads to delay themselves until sufficient data samples are collected. To see how this happens take a look at ode_samples in ODE worker, to understand ODE logic take a look at RK4Step.

enum class State
{
    Susceptible,
    Infected,
    Recovered
};

struct GPState
{
    // Tracking output data and if the GP has been trained (the latter decides the first pass logic on the ODE Solver)
    std::chrono::high_resolution_clock::time_point t0_shared;        // you'll see the vision for this in mc_worker thread
    std::chrono::high_resolution_clock::time_point t_now_shared;     // Best way to store t_now in a shared state to pass between threads while leaving it uninitialized until t_now in ode_worker is innitialized.
    std::optional<std::chrono::system_clock::time_point> maybe_time; // used to check if t_now_shared has data in it you will see it in action in the mc_worker thread and the ode worker thread
    // std::atomic<bool> advanceSimulationStageRequest{false};       // used to control issues where threads launch only once but never update and rendering loop of SFML takes over, never letting threads be called in the program
    float t_pred = 0.0f;
    float mean_I = 0.0f; // The Mean value of the normalised function that the GP is supposed to have g(t) = I(t)/N
    float var_I = 0.0f;  // The  variance of the I(t)/N at the given datapoint
    bool ready = false;  // Turns true if the GP has been trained through the Negative Logarthmic Marginalization Likelihood method based on given data

    // allow tracking of GP input data due to seg fault errors being encounterd when obsX and obsY (true values as in GPs.h) were stuck in gp_worker thread, introduce a mutex to allow their movement.
    int ObsX_rows = 0;
    int ObsY_rows = 0;
    bool Obs_valid = false;
    float last_meanI = 0.0f;
    std::mutex state_mutex;
};

GPState gp_state; // shared Global State of GP
std::mutex gp_mutex;
struct Person
{
    bool NeedtoUpdate;
    State state = State::Susceptible;
    float immuneFactor = 0.0f;
    float infectedProb = 0.0f; // Will be handled with Gaussian Process imported from GitHub
    sf::CircleShape shape;

    Person(float x, float y, float radius = 5.f)
    {
        immuneFactor = static_cast<float>(rand()) / static_cast<float>(RAND_MAX); // default immunity as a random float between 0 and 1, unsure if this applies to one object or the whole class
        shape.setRadius(radius);
        shape.setPosition(sf::Vector2f(x, y));
        shape.setFillColor(sf::Color::White);
    }
    void updateColor()
    {
        switch (state)
        {
        case State::Susceptible:
            shape.setFillColor(sf::Color::White);
            break;
        case State::Infected:
            shape.setFillColor(sf::Color::Red);
            break;
            // case State::Recovered:
            // shape.setFillColor(sf::Color::Green);
            // break;
        }
    }
};

// Shared simulation data
std::vector<Person> population; // Create an array that is able to hold multiple data types thus vector, and vectors do not specific size
std::mutex population_mutex;    // Control variable to keep data integrity during multithreading approach.
std::atomic<bool> simulationRunning(true);

// ODE solving method that I chose was Runge-Kutta 4th order
void RK4step(double &S, double &I, double Beta, double n, double mu, float P_I, double time_step)
{
    // The RHS functions according to RK4 setup
    auto dSdt = [&](double S)
    { return ((n * N) - (mu * S) - (Beta * P_I * S * (I / 144))); };
    auto dIdt = [&](double I)
    { return (Beta * P_I * S * (I / 144)); }; //- (mu * I)); };

    // k1 according to RK$
    double k1S = time_step * dSdt(S);
    double k1I = time_step * dIdt(I);

    // k2
    double k2S = time_step * dSdt(S + 0.5 * k1S);
    double k2I = time_step * dIdt(I + 0.5 * k1I);

    // k3
    double k3S = time_step * dSdt(S + 0.5 * k2S);
    double k3I = time_step * dIdt(I + 0.5 * k2I);

    // k4
    double k4S = time_step * dSdt(S + k3S);
    double k4I = time_step * dIdt(I + k3I);

    // Updating the values of S and I
    S += (k1S + 2 * k2S + 2 * k3S + k4S) / 6.0;
    I += (k1I + 2 * k2I + 2 * k3I + k4I) / 6.0;
}
// Setting up the valid function call from library files

std::mutex ode_mutex;
std::vector<std::pair<double, double>> ode_samples; // supposed to store (t, I(t)/N) which will be of use later.
std::vector<double> S_values;                       // used for thread control in main()
// Worker thread: ODE solver placeholder
void ode_worker(std::atomic<bool> &running)
{
    int iterations = 0; // for use while debugging
    double S = 143;     // initial conditions for a population of 144
    double I = 1;
    float Beta = 3;
    float n = 5;
    float mu = 5;
    bool useP_I_init;
    float P_I_init = 1.0; // First guy NEEDS to be infected or else no epidemic lol
    {
        std::lock_guard<std::mutex> lock(gp_state.state_mutex);
        useP_I_init = not(gp_state.ready); // if the GP is trained yet or not. False = P_I_init deployed
    }
    printf("\n[ODE] Starting...\n");
    ode_ready = true;                                    // Signal ready to avoid thread race
    auto t0 = std::chrono::high_resolution_clock::now(); // take starting time of the thread upon first open as the TRUE start time of the virus outbreak and set it to current system clock time
    while (running && ode_ready)
    {
        iterations++;
        {
            std::lock_guard<std::mutex> lock(gp_state.state_mutex);
            useP_I_init = not(gp_state.ready); // repeated here as it seems that use_P_init becomes true and never updates after the first iteration, thus we always use P_init if we do not have this line here
        }
        printf("\n[ode] loop:%d\n", iterations);
        auto t_now = std::chrono::high_resolution_clock::now();
        float t = std::chrono::duration<float>(t_now - t0).count(); // keeping track of current time maybe useful later
        {
            std::lock_guard<std::mutex> lock(gp_state.state_mutex);
            gp_state.t0_shared = t0;
            gp_state.t_now_shared = t_now;
            gp_state.maybe_time = std::chrono::high_resolution_clock::now(); // if this value is initialized then that means t_now_shared holds a value, this will come in handy in the mc_worker thread, I hope you see the logic because I find it difficult to explain
        }
        double RK4t_Step = 0.01; // Setting RK4StepSize this will also increase number of samples taken letting the samplePoints() logic of GPs.h become obsolete also this is related to NLML math obtaining more than 10 points for the Kernel (Covariance matrix) such that is doesn't become singular which will throw an error of datatype from the Eigen libraries used
        // Calling Runge-Kutta-4th order approach to ODE solving
        if (useP_I_init == true)
        {
            RK4step(S, I, Beta, n, mu, P_I_init, RK4t_Step);
            printf("\nThe number of Susceptible now are:%lf\n", S);
            printf("\nThe number of Infected now are:%lf\n", I);
            printf("\nThe number of ODE_worker iterations are:%d", iterations);
        }
        else
        {
            RK4step(S, I, Beta, n, mu, pInfect, RK4t_Step);
            printf("\nThe number of Susceptible now are:%lf\n", S);
            printf("\nThe number of Infected now are:%lf\n", I);
            printf("\nThe number of ODE_worker iterations are:%d", iterations);
        }
        float fracI = I / 144;
        {
            std::lock_guard<std::mutex> lock(ode_mutex);
            ode_samples.emplace_back(t, fracI);
            S_values.emplace_back(S);
            printf("\nThe number of ODE samples are:%d\n", ode_samples.size());
            if (ode_samples.size() >= 10)
            {
                ode_data_condvar.notify_all(); // Signal the gp_worker to wake up now instead of thread racing
            }
        }
        // Update ODE simulation here with locking if modifies shared state
        printf("\nSleeping for ode_worker starts\n");
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
        printf("\nSleeping for ode_worker stops\n");
    }
}

// Worker thread: GP sampling placeholder. Generates GP samples, updates shared probs with mutex
void gp_worker(std::atomic<bool> &running)
{
    static int resize_count = 0;
    static int last_N1_fitted = 0;
    int iteration_gp = 0;
    std::cout << "\ngp_worker starts\n"
              << std::endl;
    using Matrix = Eigen::MatrixXd;
    using Vector = Eigen::VectorXd;
    MatrixXd obsX; // Xd stands for X dimensions, which is determined by Eigen library itself
    MatrixXd obsY;
    RBF kernel; // Initialize a RBF covariance kernel and assign it to the model
    {           // YES I know this lock is irrelevant if you have gone through thre README on the fitModel() issue, but I kept this here so you can appreciate my thought process, you will also notice I commented out the fitModel() so you can test and see for yourself the issue about with this fitModel() thing
        std::lock_guard<std::mutex> lock(gp_mutex);
        kernel.setNoise(0.5);
        kernel.setScaling(1.0);
        gp.setKernel(kernel);
        gp.setNoise(0.5);
    }
    // bool isTimeDependantOnly = true; // Required for GP Shenaningans This will determine something important to the GP and NLML optimization, also as in out SIR model we have not accounted for Age or physical fitness so we only have time dependancy
    {
        std::lock_guard<std::mutex> lock(gp_mutex);
        gp.setSolverRestarts(1); // tells the LBFGS++ optimizer (inside fitModel()) to run the NLML minimization 0+1=1 times from different random starting points
    }
    gp_ready = true;
    while (running && gp_ready)
    {
        iteration_gp++;
        printf("\n [gp_worker] loop:%d\n", iteration_gp);
        std::vector<std::pair<double, double>> local_samples; // Defines a vector to take (t, I(t)/N) values from the ODE worker without runtime issues.
        {
            std::unique_lock<std::mutex> lock(ode_mutex); // Take data from previous mutual exclusion
            ode_data_condvar.wait(lock, [&]()
                                  { return ode_samples.size() >= 10; });
            std::cout << "\n[gp_worker] ode_samples.size() = " << ode_samples.size() << "\n";
            local_samples = ode_samples; // successfully copying data of sample data points, NLML algorithm requires atleast 10, if not it will throw runtime errors like "assertion failed index >= 0 ** index < size() ./eigen-3.3.7/Eigen/Core/DenseCoeffsBase.h line 425"
            // Debug stuff for some runtime errors:
            std::cout << "\n [gp_worker] local_samples.size() = " << local_samples.size() << "\n";
            if (local_samples.size() < 10)
            {
                ode_data_condvar.wait(lock, [&]
                                      { return ode_samples.size() >= 10; });
            }
            // Debug stuff to check if data is copied wrong due seg faults when passing GP obsX value popping up during testing remove if you see segfaults in line 73 and 509 of GPs.h
            // std::cout << "local_samples.size()=" << local_samples.size() << std::endl;
            // for (int i = 0; i < std::min(5, N1); ++i) {
            // std::cout << "sample[" << i << "]: t=" << local_samples[i].first<< " fracI=" << local_samples[i].second << std::endl;}
        }
        {
            const auto sz = local_samples.size();
            std::cout << "\nGP iteration:" << iteration_gp << ", local_samples.size()=" << sz << "\n";
        }
        if (local_samples.size() >= 10)
        {
            ++resize_count;
            int N1 = static_cast<int>(local_samples.size()); // This is not the same as N popuplation this is a special index that I do not have a name for but it is required as a count control and to manage dataset size and validation for GP
            if (resize_count <= 10)
            {
                obsX.conservativeResize(N1, 1); // dynamic resizing, because because matrix size varies each iteration also because it's cool! (☞ﾟヮﾟ)☞ . Furthermore, note the difference between obsX and gp_state.ObsX(), one is native to this worker and another is a data structure that allows for the transfer of data values
                obsY.conservativeResize(N1, 1);
                printf("\nThe size of obsX is:%d\n", obsX.size());
            }
            else
            {
                obsX.conservativeResize((N1 / 10 * resize_count), 1);
                obsY.conservativeResize((N1 / 10 * resize_count), 1);
                printf("\nThe size of obsX is:%d\n", obsX.size());
            }
            for (int i = 0; i < N1; ++i)
            {
                float t = local_samples[i].first;           // the current time at the moment the ODE is being solved is recorded here, the reason we are using current time is because at this point we still do not have I(t)/N but we can build it using these steps. Then we can move on Monte Carlo random sampling and state updates of each object of class person
                float fracI = local_samples[i].second;      // This here is just I(t)/N
                if (N1 >= 10 && (N1 - last_N1_fitted >= 5)) // Throttle GP fitting to avoid redundant optimizer work and improve stability by fitting only after enough new data accumulates
                {
                    std::cout << "\nFilling obsX: i=" << i << ", size=" << obsX.size() << std::endl;
                    obsX(i, 0) = static_cast<double>(t);
                    obsY(i, 0) = static_cast<double>(fracI);
                }
            }
            // Debug stuff to  make sure that the Matrices are of the right form of data --dealing with segfault errors in line 73 and 509 of GPs.h
            // std::cout << "=== GP DEBUG ===" << std::endl;
            // std::cout << "obsX.rows()=" << obsX.rows() << " cols=" << obsX.cols() << std::endl;
            // std::cout << "obsY.rows()=" << obsY.rows() << " cols=" << obsY.cols() << std::endl;
            // std::cout << "N1=" << N1 << std::endl;
            assert(obsX.rows() == N1 && obsX.cols() == 1);
            assert(obsY.rows() == N1 && obsY.cols() == 1);
            {
                std::lock_guard<std::mutex> lock(gp_mutex);
                std::cout << "\n Calling setObs, N1=" << N1 << "\n";
                try // I put this here because of LGFGS++ exceptions when fitModel() runs, these exceptions kill the threads, this try model will allow the thread to live on by skipping the GP update if an exception in LBFGS++ library files occurs. What type of exception you ask? All I know is that the optimizer is failing to make progress on the negative log marginal likelihood with the data and hyperparameters it sees, so it hits max_linesearch given in line 261 of LineSearchNocedalWright.h and raises an exception.
                // However I ended up throwing out fitModel() due to errors explained in the README file and adopted fixed hyperparameters instead
                {
                    // RBF kernel;
                    kernel.setNoise(0.5);
                    kernel.setScaling(1.0);
                    gp.setKernel(kernel);
                    gp.setNoise(0.5);
                    gp.setObs(obsX, obsY); // This will let us set the first input into GP
                    std::cout << "\n Calling fitModel\n";
                    // gp.fitModel(); // This function will rely on the values of obsX and obsY and the internal structure of the GP which uses RBF based Kernel and NLML based minimization for hyperparameter optimization which means it will optimize noise and scaling by itself
                    // GaussianSkip = false;         //After discovering unique_lock GaussianSkip became redundant
                }
                catch (const std::exception &e)
                {
                    std::cerr << "\n[gp_worker] fitModel failed: " << e.what() << "\n";
                    // option 1 in case you want this  it causes CRITICAL  errors but I left it in anyway so you see what I was thinking L: Reset GP completely after failure
                    //{
                    // std::lock_guard<std::mutex> lock(gp_mutex);
                    // gp = GP::GaussianProcess();   // Fresh GP object
                    // RBF kernel;
                    // kernel.setNoise(0.5); kernel.setScaling(1.0);
                    // gp.setKernel(kernel); gp.setNoise(0.5);
                    // gp.setSolverRestarts(1);
                    //}
                    // GaussianSkip - true; // skip GP until more data, made redundant by unique_lock which is objectively better
                    continue;
                }
                // Retrieve the tuned/optimized kernel hyperparameters after fitting
                auto optParams = gp.getParams();
                auto noiseL = gp.getNoise();
                auto scalingL = gp.getScaling();
            }
            last_N1_fitted = N1;
            {
                // set the means to transfer data across threads for input values of ObsX annd ObsY (t and I(t))
                std::lock_guard<std::mutex> lock(gp_state.state_mutex);
                gp_state.ObsX_rows = obsX.rows();
                gp_state.ObsY_rows = obsY.rows();
                gp_state.Obs_valid = (gp_state.ObsX_rows > 0 && gp_state.ObsY_rows > 0 && gp_state.ObsX_rows == gp_state.ObsY_rows);
                gp_state.ready = true;
                {
                    std::lock_guard<std::mutex> lock(gp_mutex);
                    gp_state.ready = true;
                    gp_data_condvar.notify_all(); // Wake mc_worker
                }
                // Debug line
                // std::cout << "GP updated: rows=" << gp_state.ObsX_rows << std::endl;
            }
        }
        std::cout << "\nSleeping for gp_worker starts\n"
                  << std::endl;
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
        std::cout << "\nSleeping for gp_worker stops\n"
                  << std::endl;
    }
}

float GaussianProcessProbI(float t_query)
{
    std::mutex FuncControl;
    int obs_rows; // The latter will define a maximum top limit of the number of samples we will take from the I(t) graphs produced
    bool valid;
    bool gp_ready;
    {
        std::lock_guard<std::mutex> lock(gp_state.state_mutex);
        obs_rows = gp_state.ObsX_rows;
        valid = gp_state.Obs_valid;
        gp_ready = gp_state.ready;
    }
    if (!valid || obs_rows == 0) // Debugging that will not let the GP run if the Data has not been fit properly or not passed properly
    {
        std::cerr << "\nGP not ready (rows=" << obs_rows << ")\n"
                  << std::endl;
        isStillSampling = true;
        {
            std::lock_guard<std::mutex> lock(FuncControl);
            AlterRetP = 50; // 5 times regular sleep time of ode_worker to obtain at least 5 ode_samples, randomly chosen. Why? "It just works" -- Todd Howard
        }
        return AlterRetP; // returning a stub time value to let the thread the function is called in wait for the GP datasample to update fully, exiting the mc_worker thread whence called, because if this happens then the data has not been passed between threads.
    }
    else
    {
        std::lock_guard<std::mutex> lock(gp_mutex);
        using Matrix = GP::Matrix; // Calling in library files from CppGP, this one is a derivative from Eigen library that allows us to represent data in TRUE Matrices, which is essential to simplify the GP algorithm's computation, it is complicated to explain but you can see it in GPs.h, which can be obtained by Git pulling all the libraries in the README file
        using Vector = GP::Vector;
        Matrix predX(1, 1);                         // predX is a structure based on Eigen library files Matrix data type that stores input data points of input times
        predX(0, 0) = static_cast<double>(t_query); // usually t_query will be t_monte which will allow us to random sample!
        Matrix predMean;
        Matrix predvari;
        {
            std::lock_guard<std::mutex> lock(gp_mutex);
            gp.setPred(predX);
            gp.predict();
            predMean = (gp.getPredMean()); // This will return the predicted mean of the normal distribution of uncertainty about the value of I(t)/N at the given t_monte time which can later be used to find P(I|C)
            predvari = (gp.getPredVar());
        }
        float meanI = static_cast<float>(predMean(0, 0)); // this is I(t)/N
        float varI = static_cast<float>(predvari(0, 0));

        // Getting P(I|C) native vs mc_worker method
        float p_2 = std::max(0.0f, std::min(1.0f, 2.0f * meanI)); // This here is a "clamping" function that weighs predMean between 0 and 1 and then assigns it a weightage that is doubled or boosted (boosting is because the weightage may sometimes be too small cayusing the percolation to take too long), taking the upper maximum, I got his off of the same guy on StackOverflow,crazy what they know
        // alternatively you can use the actual probability formula in mc_worker as it needs access to t_monte
        // float p_2 = exp((pow((-1 * (t_query - meanI)), 2)) / (2 * (pow(varI, 2)))) / sqrt(3.14 * 2 * (pow(varI, 2))); // taking pi as 3.14 in Normal distribution probability be warned though the dimensions here are off as the variance is in I(t) but the mean is taken in t (I think), I tried to get it right but haven't found a way so far
        return p_2;
    }
}

// Worker thread: Monte Carlo infection sampling and state assigning, this is the final step.
void mc_worker(std::atomic<bool> &running, std::mt19937 &gen)
{
    int mc_worker_iter = 0;
    std::cout << "\nmc_worker starts\n"
              << std::endl;
    std::uniform_real_distribution<float> probDist(0.0f, 1.0f);
    float t_monte;
    float pI;
    while (running)
    {
        if (!ode_ready.load())
        {
            std::this_thread::sleep_for(std::chrono::milliseconds(10));
            continue;
        }
        mc_worker_iter++;
        printf("\n [mc_worker] loop:%d\n", mc_worker_iter);

        // wait for GP data instead of GaussianSkip
        {
            std::unique_lock<std::mutex> lock(gp_mutex);
            gp_data_condvar.wait(lock, [&]
                                 { return gp_state.maybe_time.has_value() && gp_state.ready; }); // I discovered this recently, this allowed me to rewrite uncomplicate most of my logic, I will still leave the old logic here for your reference
        }
        // if (GaussianSkip == true) // if the GP Process has been skipped because there aren't enough datapoints from the ODE_thread yet, then we must wait until there are enough datapoints
        //{
        // std::this_thread::sleep_for(std::chrono::milliseconds(50));
        // continue;
        //}
        // read once to center the calculation, got this tip off of stackoverflow, without this my calculations were off by a constant factor
        //{
        // std::lock_guard<std::mutex> lock(gp_state.state_mutex);
        // if (!gp_state.maybe_time.has_value()) // if the ODE worker has not yet set t_now and the GPState and not caught it yet then wait a small amount of time for it to fill up then continue
        //{
        // std::this_thread::sleep_for(std::chrono::milliseconds(10));
        // continue;
        //}
        auto t_now_mc_worker = gp_state.t_now_shared;
        auto t0_mc_worker = gp_state.t0_shared;
        float t_now_float_type = std::chrono::duration<float>(t_now_mc_worker - t0_mc_worker).count(); // float t = std::chrono::duration<double>(t_now - t0).count();
        std::uniform_real_distribution<float> timeDist(0.0f, t_now_float_type);
        t_monte = timeDist(gen);            // already in [0, t_now]
        pI = GaussianProcessProbI(t_monte); // from GP posterior without actually needing to output a plot on .csv or .xsl file
        while (pI == 50)
        {
            std::this_thread::sleep_for(std::chrono::milliseconds(50));
            pI = GaussianProcessProbI(t_monte);
        }
        if ((pI >= 0) && (pI <= 1))
        {
            pInfect = Beta * pI;
        }
        else
        {
            printf("Unknown error, you're screwed.");
        }

        // pInfect = std::max(0.0f, std::min(1.0f, pInfect)); // clamp in case you wish to use alternate p not p_2 but the one suggested from stackoverflow

        {
            std::lock_guard<std::mutex> lock(population_mutex);
            for (auto &person : population)
            {
                if (person.state == State::Susceptible)
                {
                    float adjustedProb = pInfect * (1.0f - person.immuneFactor);
                    adjustedProb = std::max(0.0f, std::min(1.0f, adjustedProb));

                    if (probDist(gen) < adjustedProb)
                    {
                        person.NeedtoUpdate = true;
                        person.state = State::Infected;
                        person.updateColor();
                        // temp debug
                        printf("\n Infecting a new person with adjustedProb:%lf\n", adjustedProb);
                    }
                }
            }
        }
        std::cout << "\nSleeping for mc_worker starts\n"
                  << std::endl;
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
        std::cout << "\nSleeping for mc_worker stops\n"
                  << std::endl;
    }
}

int main()
{
    int S_local_main_prev;
    const int windowWidth = 193, windowHeight = 218;
    const int numCols = 12, numRows = 12;
    const float radius = 8.f, spacing = 0.f;
    double S_local_main;
    bool First_iter = false;

    sf::RenderWindow window(sf::VideoMode(sf::Vector2u(windowWidth, windowHeight)), "Epidemic Simulation");
    window.setFramerateLimit(60);

    // Set up the popultation Grid
    for (int i = 0; i < numRows; ++i)
    {
        for (int j = 0; j < numCols; ++j)
        {
            {
                std::lock_guard<std::mutex> lock(population_mutex);
                population.emplace_back(j * (2 * radius + spacing), i * (2 * radius + spacing), radius);
            }
        }
    }
    {
        std::lock_guard<std::mutex> lock(population_mutex);
        // preset 1 guy to be infected already to start the epidemic or else we won't have an epidemic and all circles will be white, I know this from debugging experience
        population[0].state = State::Infected;
        population[0].updateColor();
    }

    // Setup random generator seeded by random_device and using Marsenne Twister method for more random psuedo-random numbers
    std::random_device rd;
    std::mt19937 gen(rd());

    // Clearing states to prepare launching threads, letting us get rid of junk data
    {
        std::lock_guard<std::mutex> lock(ode_mutex);
        ode_samples.clear();
        S_values.clear();
    }
    ode_ready = false;
    gp_ready = false;
    // Launching the threads for ODE, GP and Monte Carlo
    simulationRunning = true;
    std::thread odeThread(ode_worker, std::ref(simulationRunning));
    std::thread gpThread(gp_worker, std::ref(simulationRunning));
    std::thread mcThread(mc_worker, std::ref(simulationRunning), std::ref(gen));
    // printf("\n Good AF\n"); used in debugging to check if I reached all threads to start
    {
        std::lock_guard<std::mutex> lock(ode_mutex);
        if (!S_values.empty())
        {
            S_local_main = S_values.back();
            S_local_main_prev = (int)S_local_main;
        }
    }

    // The rendering of all of our work in SFML
    while (window.isOpen())
    {
        while (const auto event = window.pollEvent())
        {
            if (event->is<sf::Event::Closed>())
            {
                window.close();
            }
            else if (const auto *keyPressed = event->getIf<sf::Event::KeyPressed>())
            {
                if (keyPressed->scancode == sf::Keyboard::Scancode::Escape)
                {
                    window.close();
                }
            }
        }
        // window.clear(sf::Color::Black);
        static sf::Clock simClock;
        if (simClock.getElapsedTime().asMilliseconds() > 50)
        {
            // Update S_local_main based on current samples and count used to allow for a termination condition
            {
                std::lock_guard<std::mutex> lock(ode_mutex);
                if (!S_values.empty())
                {
                    S_local_main_prev = (int)S_local_main;
                    S_local_main = S_values.back();
                    First_iter = false;
                }
            }
        }

        // Updating our window
        window.clear(sf::Color::Black);
        {
            std::lock_guard<std::mutex> lock(population_mutex);
            int counter;
            auto person_count = population.size();
            // for (auto &person : population)    //old loop idea
            for (counter = 0; counter < person_count; ++counter)
            {
                if (population[counter].NeedtoUpdate == true)
                {
                    population[counter].state = State::Infected;
                    population[counter].updateColor();
                    population[counter].NeedtoUpdate = false; // will become true again when mc_wotker keeps running
                    printf("\nUpdating colour scheme of person number:%d", counter);
                }
                window.draw(population[counter].shape);
                // window.draw(person.shape); //old logic for old for loop
            }
        }
        window.display();

        if (((S_local_main == 0) && (((S_local_main - S_local_main_prev == 0) && (!First_iter))) || ((S_local_main - S_local_main_prev == 0) && (!First_iter))))
        {
            // choose if you want to stop sim threads here:
            simulationRunning = false;
        }
    }

    // joining all threads and closing the window
    simulationRunning = false;
    odeThread.join();
    gpThread.join();
    mcThread.join();
    return 0;
}