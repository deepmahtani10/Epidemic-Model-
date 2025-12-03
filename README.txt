For this experimental project I wanted to create an SIR model that coupled Gaussian Processes with Monte Carlo Random Sampling and SFML with multithreading. I gcc 15.2.0 to compile SFML from source here.

If you're new to SFML (like I was about 1 month ago) here's a youtube tutorial, SFML is a user-friendly library for 2D graphics, windowing, and input. It’s great for drawing shapes, text, and managing graphics in real-time simulations.
You can get SFML for free here: https://www.sfml-dev.org/download/
Be warned that using gcc requires that your MinGW and UCRT64 system architecture match up with the ones shown on the download page. 
Do note that SFML pre-compiled versions are not compatible with gcc 15.2.0 (The version I used)

I'd like to add that starting 01/11/2025 I began to learn SFML from scratch, so if you see any inefficient code do let me know how I can make it better. 

The in-built ibraries I have used are <chrono>, <atomic>, <mutex>, <thread>, <vector>, <random>, <condition_variable>, <iostream> (for some reason)
Some external libraries I used are CppGp, Eigen, LBFGS++ and boost which can all be found here https://github.com/nw2190/CppGPs, here https://libeigen.gitlab.io/eigen/docs-nightly/GettingStarted.html, there https://github.com/yixuan/LBFGSpp and yonder https://www.boost.org/releases/latest/  

My multithreading idea goes like this:

+-----------------+          +------------------+
|    Main Thread  | <--read--| Worker Threads   |
| - SFML window   |          | - ODE solver     |
| - Event poll    |          | - GP sampling    |
| - Draw scene    |          | - Monte Carlo    |
+-----------------+          +------------------+
           |                         |
       update shared state variables (mutex/atomic)


This approach was decided on because my epidemic model follows idea called the SI (Susceptible-Infected) version of the SIR model (Susceptible-Infected-Recovered model) which allows me to find the number of people Susceptible (not infected but can get infected) S(t), the number of Infected people I(t) and, the number of Recovered people R(t) using a system of 3 ODEs shown below:
The reason I only use SI instead of SIR is because I am only concerned with the transmission and percolation of the virus in the epidemic's affected population.
dS(t)/dt = nN - μ*S(t) - [β*Prob(Infected given contact with another person)*Prob(Susceptible given contact with another person)*I(t)]/N
dI(t)/dt = [β*Prob(Infected given contact with another person)*Prob(Susceptible given contact with another person)*I(t)*S(t)/N] - μ*I(t)
dR(t)/dt = γI(t) - μR(t)

where:
n = number of births
μ = number of deaths 
n = μ = 5 for this case, why? Because it's a random number and the Gaussian Process model is just too much work sometimes and I really want to simplify where I can without changing too much of the problem, this is one place where I can do that 
N = Total population (fixed meaning n = μ !!!) and N = 144.
β = Average number of contacts to be infected (fixed in this case but varies in the real world. For example coming across someone who has the flu 1 time may not always get you sick, and modelling this dynamically is a big headache)

For some information on the SIR model you can follow these links: 

https://www.youtube.com/watch?v=bkEAWJwGFBQ              (Lecture from IIT India)
https://www.youtube.com/watch?v=IXkr0AsEh1w		         (Lecture from National Research Uniersity Higher School of Economics, location: unkown to me)
https://ieeexplore.ieee.org/abstract/document/7272972    (using SIR models to see epidemics in browser games)
https://academic.oup.com/jid/article/212/9/1420/1025422  (For further reading on highly robust interpretations of the SIR model that follow biology more accurately than my humble project here)

Things you may need to know about the development process when I was handling the ODEs

UPDATE 1 15/11/2025 : I opted to go for the simple version of dI(t)/dt = β*P(Infected given Contact)*I(t) and dS(t)/dt = nN - μ*S(t). Though there may be a slight difference in accuracy, (I soon changed it however to the version you'll see in Main.cpp)
UPDATE 2 27/11/2025 : I noticed that with numbers N=144, S=143 and I=1 with β = 0.5, P_I_init = 1.0 (this you will see in the code itself), n = 5 and mu = 5, the initial Infection magnitude =0.497 -mu*I = 0.497 - 5 = -4.503 therefore infection decreases. Hence I am dropping the -muI term in my program for the given N
UPDATE 3 27/11/2025 : I noticed that the return value p_2 of GuassianProcessProbI(t_query){} is a Gaussian Probability Distribution Function that uses the numerator -((t_query - mu)^2) which is dimensionally mismatched from the denominator variance. As the numerator is a distance in the time axis but the denominator is defined on the infected-fraction axis. Thus I commented it out and used a clamp version. The Gaussian process is still Gaussian nature though because the GP is already trained on the point (t_monte, I(t_monte)/N) producing many Ansatz fitted I(t).

The overall plan is to simulate an epidemic on N people, represented as White Circles, arranged in a 2D grid is given as follows:

1.Initialise N population as a grid of White circles just touching each other, each circle represets an object of class person.
2.The Class person is declared with attributes ImmmuneFactor (a random float value between 0 and 1) and State (Susceptible, Infected or Recovered) and infectionProbability 
3.I(t) and S(t) are found from the system of ODEs dI(t)/dt = β*P(I|C)*I(t) and dS(t)/dt = nN - μS(t)
4.From Prob Gaussian_Process(I(t)) = P(I|C) for every unit of time we obtain a g(t) = P(I|C)
5.If we Random sample g(t) within a t interval no greater than the total time elapsed till this step from the start of the program (meaning we pick a random value for t that less than or greater than the total time elapsed using a Mersenne Twister random process). This t value is random sampled through the following lines
 // Setup random generator seeded by random_device
    std::random_device rd;
    std::mt19937 gen(rd());
 //The part below is within the worker thread
    while (running) 
        float t_current = timeDist(gen);
        float pInfect = Beta * GaussianProcessProbI(t_current);
6.Compare I(t_current)/N < infectedProb IF TRUE, then State is updated to infected

Lastly P(I|C) for a given time and person is assumed to be same as P(I(t_specific/N)) obtained the Gaussian Process of modelling I(t)/N to obtain the normal distribution of uncertainty about every data point

You may want to watch out for the following errors: The issue of the GP working and ODE solver working and MC random sampling working and the multithreading working but due to scaling problems and numerical instability inherent in the NLML optimization of the GP we get runtime errors that cause the program to fail.
                                                    These errors arise when kernel hyperparameters cause the covariance matrix to be near singular

I had to modify the GPs.h file a bit to include new Get Methods in the constructor GuassianProcess called getObsXRows(), getObsYRows() and isObsValid() to ensure data passed between all threads did not result in a seg fault
This resulted in me also changing GPs.cpp. I will link all modified files in my ZIP but it will be a very tedious task to put it all together because of the placement of all files

Do note another issue I encountered in the GP hyperparameter optimization phase: The line search and gradient method coded into the GP library files seemed to fail whenever at times, for reasons still unknown to me, they would return NULL values corrupting the whole obsX and obsY Matrices in gp_worker.
This corruption also triggers errors in the Eigen-library as a out-of-bounds access to some variable since everything turns NULL and the size of the matrices is a NULL value, creating an avalanche of problems.
I tried debugging by messing with the .h files, which gave little fruit to my efforts, I also tried throttling of fitting frequency by not trying to fit the model in regular intervals and i tried incremental resizing of the data matrices given by resize_count mechanics, these had seemed to work but sometimes this issue persisted.

one of the changes to GPs.h I tried to make was the following:
void GaussianProcess::reset() {
    obsX.setZero();
    obsY.setZero();
}

it seems like a good restart point when data gets corrupted. Yet, using this would mean that we have to start the GP process all over again. On the other hand, it does not stop the corruption errors from happening again,if invoked there is the very real possibility of ending up in an infinite loop of resets whenever this hyperparameter optimization issue occurs.

So a quick and dirty fix I did wsa to just throw in some hyperparameters myself and not call fitModel() to optimize them. In other words I pretended that these hyperparameters were pre-optimized and had the GP run on these values

Below is a sample of the terminal output during debugginng:

he number of ODE_worker iterations are:22
The number of ODE samples are:22

Sleeping for ode_worker starts

Sleeping for ode_worker stops

[ode] loop:23

The number of Susceptible now are:143.069463

The number of Infected now are:1.985081

The number of ODE_worker iterations are:23
The number of ODE samples are:23

Sleeping for ode_worker starts

Sleeping for ode_worker stops

[ode] loop:24

The number of Susceptible now are:143.057136

The number of Infected now are:2.045139

The number of ODE_worker iterations are:24
The number of ODE samples are:24

Sleeping for ode_worker starts

Sleeping for ode_worker stops

[ode] loop:25

The number of Susceptible now are:143.043669

The number of Infected now are:2.107009

The number of ODE_worker iterations are:25
The number of ODE samples are:25

Sleeping for ode_worker starts

Sleeping for ode_worker stops

[ode] loop:26

The number of Susceptible now are:143.029066

The number of Infected now are:2.170744

The number of ODE_worker iterations are:26
The number of ODE samples are:26

Sleeping for ode_worker starts

Sleeping for ode_worker stops

[ode] loop:27

The number of Susceptible now are:143.013330

The number of Infected now are:2.236401

The number of ODE_worker iterations are:27
The number of ODE samples are:27

Sleeping for ode_worker starts

Sleeping for ode_worker stops

[ode] loop:28

The number of Susceptible now are:142.996461

The number of Infected now are:2.304036

The number of ODE_worker iterations are:28
The number of ODE samples are:28

Sleeping for ode_worker starts

Sleeping for ode_worker stops

[ode] loop:29

The number of Susceptible now are:142.978456

The number of Infected now are:2.373708

The number of ODE_worker iterations are:29
The number of ODE samples are:29

Sleeping for ode_worker starts

Sleeping for ode_worker stops

[ode] loop:30

The number of Susceptible now are:142.959315

The number of Infected now are:2.445477

The number of ODE_worker iterations are:30
The number of ODE samples are:30