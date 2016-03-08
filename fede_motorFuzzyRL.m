%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%
%%% SC4081 Knowledge Based Control Systems
%%%
%%% DC Motor Controlled with Fuzzy Reinforcement Learning Controller
%%%
%%% J. Lee (4089286), I. Matamoros (4510704), F. Paredes (4439953) and L. Valk (4095154)
%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%
%%% System Dynamics (Model from Busoniu et. al., 2010.)
%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%% DC motor system matrices 
A = [1 0.0049; 
     0 0.9540];
 
B = [0.0021;
     0.8505];

%%% Sampling time
Ts = 0.005;

%%% LQR Control matrices
Qcost = [5 0;
     0 0.01]; 
Rcost = 0.01;

Ncost = 0;

%%% LQR feedback gain
% F = dlqr(A,B,Qcost,Rcost,Ncost);

%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%
%%% Reinforcement learning. Since the model is known, this can be done
%%% offline without making observations from a simulation.
%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%
%%% Model state definitions
%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Define boundaries of the signals
alpha_bounds = [-pi    pi];
omega_bounds = [-16*pi 16*pi];
u_bounds     = [-10    10];

nControlSteps = 15;
u_values     = linspace(    u_bounds(1),    u_bounds(2),nControlSteps);

gamma = 0.95;
eps = 1;
it = 0;

%%% Choose which q iteration we do
do_fuzzy_q_iteration = 1;

%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%
%%% Fuzzy Q iteration
%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

if(do_fuzzy_q_iteration)
    
    % Define number of triangular fuzzy partitions
    nAlphaTriang = 41;
    nOmegaTriang = nAlphaTriang;
    
    % Initialize Theta matrices
    Theta0 = zeros(nAlphaTriang,nOmegaTriang,nControlSteps);
    Theta  = Theta0;
   
    % Define the cores of the membership functions
    alpha_triangles = linspace(alpha_bounds(1),alpha_bounds(2),  nAlphaTriang);
    omega_triangles = linspace(omega_bounds(1),omega_bounds(2),  nOmegaTriang);
        
    
    while (eps > 1e-8)
        it = it + 1;
        Thetanext = Theta0;
        for alpha_index = 1:nAlphaTriang
            for omega_index = 1:nOmegaTriang
                for u_index = 1:nControlSteps
                   % Do the update for each element

                   xNow = [alpha_triangles(alpha_index);
                           omega_triangles(omega_index)];

                   uNow = u_values(u_index);

                   xNext = A*xNow + B*uNow;
                   
                   if(xNext(1) > alpha_bounds(2))
                       xNext(1) = alpha_bounds(2);
                   elseif xNext(1) < alpha_bounds(1)
                           xNext(1) = alpha_bounds(1);
                   end
                   
                   if(xNext(2) > omega_bounds(2))
                       xNext(2) = omega_bounds(2);
                   elseif xNext(2) < omega_bounds(1)
                           xNext(2) = omega_bounds(1);
                   end                       
                     

                   alphaNext = xNext(1);
                   omegaNext = xNext(2);

                   rewardNow = -xNow'*Qcost*xNow - uNow'*Rcost*uNow;        

                   SumdotMultiply = zeros(1,nControlSteps);  

                   for u_prime = 1:nControlSteps
                       
                       [phi_alpha_vector, phi_omega_vector] = fede_MF(xNext, nAlphaTriang, nOmegaTriang, alpha_bounds, omega_bounds);


                       MuMatrix = min(  repmat(phi_omega_vector',[nAlphaTriang 1]) , repmat(phi_alpha_vector,[1 nOmegaTriang])  );
                       
                       
                       SumdotMultiply(u_prime) = sum(sum(MuMatrix.*Theta(:,:,u_prime)));

                    end
                    
                    [MuThetavalsMax,MuThetavalsMaxIndex] = max(SumdotMultiply);

                   Thetanext(alpha_index,omega_index,u_index) = rewardNow + gamma*MuThetavalsMax;
                end
            end       
        end

        eps = max(max(max(abs(Theta-Thetanext))));
        disp(['Fuzzy Iteration: ' num2str(it) ' eps = ' num2str(eps)]);        
        Theta = Thetanext;  
    end    
    
%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%
%%% Conventional Q iteration
%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        
else % Do normal q iteration
    
    % Define number of discrete steps for the states and the control signal.
    nAlphaSteps = 21;
    nOmegaSteps = nAlphaSteps;

    % Define the values that the states and the control signal can attain.
    alpha_values = linspace(alpha_bounds(1),alpha_bounds(2),  nAlphaSteps);
    omega_values = linspace(omega_bounds(1),omega_bounds(2),  nOmegaSteps);    
    
    % Initialize Q matrices
    Q0 = zeros(nAlphaSteps,nOmegaSteps,nControlSteps);
    Q  = Q0;
    
    while (eps > 1e-3 && it < maxNumberOfIterations)
        it = it + 1;
        Qnext = Q0;
        for alpha_index = 1:nAlphaSteps
            for omega_index = 1:nOmegaSteps
                for u_index = 1:nControlSteps
                   % Do the update for each element

                   xNow = [alpha_values(alpha_index);
                           omega_values(omega_index)];

                   uNow = u_values(u_index);

                   xNext = A*xNow + B*uNow;

                   alphaNext = xNext(1);
                   omegaNext = xNext(2);

                   [~,alphaNextIndex] = min(abs(alpha_values - alphaNext));
                   [~,omegaNextIndex] = min(abs(omega_values - omegaNext));


                   rewardNow = -xNow'*Qcost*xNow - uNow'*Rcost*uNow;        

                   Qval = max(Q(alphaNextIndex,omegaNextIndex,:));

                   Qnext(alpha_index,omega_index,u_index) = rewardNow + gamma*Qval;
                end
            end       
        end

        eps = max(max(max(abs(Q-Qnext))));
        disp(['Iteration: ' num2str(it) ' eps = ' num2str(eps)]);
        Q = Qnext;  
    end    
end

%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%
%%% Simulation of the system, currently using a state feedback (LQR)
%%% controller. (Once reinforcement learning section above is complete, we
%%% can use the finished Q function for control instead.)
%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%% Simulation length
t_end = 0.5;
time = 0:Ts:t_end;

%%% Preparing empty matrices to store the simulation results
nStates = size(A,1);         % Number of states in the system
nInputs = size(B,2);         % Number of control inputs
nSamples = length(time);     % Number of time samples
X = zeros(nStates,nSamples); % Matrix to store the system state  at all samples
U = zeros(nInputs,nSamples); % Matrix to store the control input at all samples
reward = zeros(1,nSamples);
rewardDiscrete = zeros(1,nSamples);

%%% Initial state
X(:,1) = [-pi;
          0];

for k = 1:nSamples  
    % Control signal for this step
    xNow = X(:,k);
    
    alphaNow = xNow(1);
    omegaNow = xNow(2);  
    
    if(do_fuzzy_q_iteration)
        SumdotMultiply = zeros(1,nControlSteps);  

        for u_prime = 1:nControlSteps
            
           xNowLimited = xNow; 
           if(xNowLimited(1) > alpha_bounds(2))
               xNowLimited(1) = alpha_bounds(2);
           elseif xNowLimited(1) < alpha_bounds(1)
                   xNowLimited(1) = alpha_bounds(1);
           end

           if(xNowLimited(2) > omega_bounds(2))
               xNowLimited(2) = omega_bounds(2);
           elseif xNowLimited(2) < omega_bounds(1)
                   xNowLimited(2) = omega_bounds(1);
           end             

           [phi_alpha_vector, phi_omega_vector] = fede_MF(xNowLimited, nAlphaTriang, nOmegaTriang, alpha_bounds, omega_bounds);

           MuMatrix = min(  repmat(phi_omega_vector',[nAlphaTriang 1]) , repmat(phi_alpha_vector,[1 nOmegaTriang])  );

           SumdotMultiply(u_prime) = sum(sum(MuMatrix.*Theta(:,:,u_prime)));

        end

        [MuThetavalsMax,MuThetavalsMaxIndex] = max(SumdotMultiply);  
        u_index = MuThetavalsMaxIndex;
    else
        [~,alphaNowIndex] = min(abs(alpha_values - alphaNow));
        [~,omegaNowIndex] = min(abs(omega_values - omegaNow));  

        xDiscreteNow = [alpha_values(alphaNowIndex);
                        omega_values(omegaNowIndex)];

        [~,u_index] = max(Q(alphaNowIndex,omegaNowIndex,:));
        
    end
    
    uDiscreteNow = u_values(u_index);
    U(k) = uDiscreteNow;    
    
    
    reward(k) = -xNow'*Qcost*xNow - uDiscreteNow'*Rcost*uDiscreteNow;
    
    % State at next step
    if(k < nSamples)
        X(:,k+1) = A*X(:,k) + B*U(k);
    end
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%
%%% Plot simulation results
%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
close all;

%%% Plot the angle
subplot(4,1,1);
plot(time,X(1,:));
ylabel('\phi (rad)')

%%% Plot the angular velocity
subplot(4,1,2);
plot(time,X(2,:));
ylabel('\omega (rad/s)')

%%% Plot the control signal
subplot(4,1,3);
plot(time,U(:));
ylabel('u')

%%% Plot the reward signal
subplot(4,1,4);

plot(time,[reward(:)]);
    ylabel('Reward')
    xlabel('time (s)');  
