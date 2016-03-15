%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%
%%% SC4081 Knowledge Based Control Systems
%%%
%%% DC Motor Controlled with Fuzzy Reinforcement Learning Controller
%%%
%%% J. Lee (4089286), I. Matamoros (4510704), F. Paredes Valles (4439953) and L. Valk (4095154)
%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%
%%% Choose to load previously stored results or run the Q-iteration again.
%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

clc; clear all; 

load_previous_results = false;

if(load_previous_results)
    disp('Loading previous results for analysis')
    
    load('results/results_2016-03-14_13-55-27-397.mat') % A simple conventional Q learning result
    
%     load('results/results_fede.mat')                  % The results obtained by Fede
else
    disp('Recomputing results')
    
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

    gamma = 0.99;
    eps = 1;
    it = 0;
    
    maxNumberOfIterations = inf;

    %%% Choose which q iteration we do
    do_fuzzy_q_iteration = 0;

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


        while (eps > 1e-8 && it < maxNumberOfIterations)
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

                       rewardNow = -xNow'*Qcost*xNow - uNow'*Rcost*uNow;        

                       SumdotMultiply = zeros(1,nControlSteps);  

                       for u_prime = 1:nControlSteps

                           [phi_alpha_vector, phi_omega_vector] = MF(xNext, nAlphaTriang, nOmegaTriang, alpha_bounds, omega_bounds);

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
        nAlphaSteps = 41;
        nOmegaSteps = nAlphaSteps;

        % Define the values that the states and the control signal can attain.
        alpha_values = linspace(alpha_bounds(1),alpha_bounds(2),  nAlphaSteps);
        omega_values = linspace(omega_bounds(1),omega_bounds(2),  nOmegaSteps);    

        % Initialize Q matrices
        Q0 = zeros(nAlphaSteps,nOmegaSteps,nControlSteps);
        Q  = Q0;

        while (eps > 1e-8 && it < maxNumberOfIterations)
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

    resultfileUTCtime = datestr(datetime('now','TimeZone','UTC'),'YYYY-mm-dd_HH-MM-SS-FFF');
    resultfilename = ['results/results_' resultfileUTCtime];
    save([resultfilename]);
    disp(['Storing results in: ' resultfilename '.mat'])
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
X(:,1) = [-pi;0];

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

           [phi_alpha_vector, phi_omega_vector] = MF(xNowLimited, nAlphaTriang, nOmegaTriang, alpha_bounds, omega_bounds);

           MuMatrix = min(repmat(phi_omega_vector',[nAlphaTriang 1]) , repmat(phi_alpha_vector,[1 nOmegaTriang]));

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

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%
%%% Reconstruct fuzzy Q
%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

u_zero_index = (nControlSteps+1)/2; % To see plot for u  0;

if(do_fuzzy_q_iteration)

    nPlotPoints = 50;
    alpha_values_fuzzy_plot = linspace(alpha_bounds(1),alpha_bounds(2),nPlotPoints);
    omega_values_fuzzy_plot = linspace(omega_bounds(1),omega_bounds(2),nPlotPoints);

    fuzzyQplotvalues = zeros(nPlotPoints,nPlotPoints);
    
    for alpha_plot_index = 1:nPlotPoints
        for omega_plot_index = 1:nPlotPoints
            alpha_now = alpha_values_fuzzy_plot(alpha_plot_index);
            omega_now = omega_values_fuzzy_plot(omega_plot_index);
            xNow = [alpha_now;omega_now];
            
            [phi_alpha_vector, phi_omega_vector] = MF(xNow, nAlphaTriang, nOmegaTriang, alpha_bounds, omega_bounds);
            MuMatrix = min(  repmat(phi_omega_vector',[nAlphaTriang 1]) , repmat(phi_alpha_vector,[1 nOmegaTriang])  );
             
            fuzzyQplotvalues(alpha_plot_index,omega_plot_index) = sum(sum(MuMatrix.*Theta(:,:,u_zero_index)));
        end
    end

end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%
%%% Reconstruct policy
%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

if(do_fuzzy_q_iteration)
    
    nPlotPoints = 50;
    alpha_values_fuzzy_plot = linspace(alpha_bounds(1),alpha_bounds(2),nPlotPoints);
    omega_values_fuzzy_plot = linspace(omega_bounds(1),omega_bounds(2),nPlotPoints);
    u_values_fuzzy_plot     = zeros(nPlotPoints,nPlotPoints);

    
    counter = 0;
    
    for alpha_plot_index = 1:nPlotPoints
        for omega_plot_index = 1:nPlotPoints 
            for u_prime = 1:nControlSteps
                alpha_now = alpha_values_fuzzy_plot(alpha_plot_index);
                omega_now = omega_values_fuzzy_plot(omega_plot_index);
                xNow = [alpha_now; omega_now];

               [phi_alpha_vector, phi_omega_vector] = MF(xNow, nAlphaTriang, nOmegaTriang, alpha_bounds, omega_bounds);
                MuMatrix = min(repmat(phi_omega_vector',[nAlphaTriang 1]) , repmat(phi_alpha_vector,[1 nOmegaTriang]));
                SumdotMultiply(u_prime) = sum(sum(MuMatrix.*Theta(:,:,u_prime)));
            end

        [MuThetavalsMax,MuThetavalsMaxIndex]            = max(SumdotMultiply);  
         u_values_fuzzy_plot(alpha_plot_index,omega_plot_index) = u_values(MuThetavalsMaxIndex);
         
        end
    end
    
end

%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%
%%% Create figures and handles
%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
close all;    

%%% Handles for the all-in-one figure
combinedFig = figure;
combinedAx_Alpha  = subplot(4,2,1);
combinedAx_Omega  = subplot(4,2,3);
combinedAx_u      = subplot(4,2,5);
combinedAx_r      = subplot(4,2,7);
combinedAx_Q      = subplot(4,2,[2,4]);
combinedAx_Policy = subplot(4,2,[6,8]);

%%% Handles for the time signal figure
separateFig_Time  = figure;
separateAx_Alpha  = subplot(4,1,1);
separateAx_Omega  = subplot(4,1,2);
separateAx_u      = subplot(4,1,3);
separateAx_r      = subplot(4,1,4);

%%% Handles for the Q value figure
separateFig_Q     = figure;
separateAx_Q      = subplot(1,1,1);

%%% Handles for the policy figure
separateFig_Policy = figure;
separateAx_Policy  = subplot(1,1,1);

%%% Properties for all figures
% set(findall(gcf,'type','text'),'FontSize',15)

%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%
%%% Plot time simulation results
%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

Width = 1.5;

%%% Plot the angle
for ax = [combinedAx_Alpha separateAx_Alpha]
    axes(ax);
    plot(time,X(1,:),'k','LineWidth',Width);
    ylabel('\phi (rad)')
    grid on
    xlim([0 t_end])
end

%%% Plot the angular velocity
for ax = [combinedAx_Omega separateAx_Omega]
    axes(ax)
    plot(time,X(2,:),'k','LineWidth',Width);
    ylabel('\omega (rad/s)')
    grid on
    xlim([0 t_end])    
end

%%% Plot the control signal
for ax = [combinedAx_u separateAx_u]
    axes(ax)
    plot(time,U(:),'k','LineWidth',Width);
    ylabel('u (V)')
    xlim([0 t_end])
    grid on
end

%%% Plot the reward signal
for ax = [combinedAx_r separateAx_r]
    axes(ax)
    plot(time,[reward(:)],'k','LineWidth',Width);
    ylabel('r (-)')
    xlabel('time (s)');  
    grid on
    xlim([0 t_end])
end

%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%
%%% Plot Q values or Q function and Policy
%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

for ax = [combinedAx_Q separateAx_Q]
    axes(ax)
    colormap('gray')
    view(0,90)
    %%% Plot the Q function for u = 0
    if(do_fuzzy_q_iteration)
        Qplotvalues = fuzzyQplotvalues;
        alpha_values_plot = alpha_values_fuzzy_plot;
        omega_values_plot = omega_values_fuzzy_plot;
    else
        Qplotvalues = Q(:,:,u_zero_index);
        alpha_values_plot = alpha_values;
        omega_values_plot = omega_values;        
    end

    surf(alpha_values_plot,omega_values_plot,Qplotvalues','EdgeColor','none');

    % xlabel('\phi (rad)')
    ylabel('\omega (rad/s)')
    title('Q(\phi,\omega,0) (-)')
    axis([-pi pi -16*pi 16*pi])
end

%%% Plot the policy
for ax = [combinedAx_Policy separateAx_Policy]
    axes(ax)   
    
    if(do_fuzzy_q_iteration)
        u_values_plot = u_values_fuzzy_plot;
        alpha_values_plot = alpha_values_fuzzy_plot;
        omega_values_plot = omega_values_fuzzy_plot;        
    else
        [~, policyIndices] = max(Q,[],3);        
        u_values_plot = u_values(policyIndices);
        alpha_values_plot = alpha_values;
        omega_values_plot = omega_values;          
    end
    
    surf(alpha_values_plot,omega_values_plot,u_values_plot','EdgeColor','none');  
    colorbar
    view(0,90)
    grid on
    colormap('gray')
    colorbar
    view(0,90)
    xlabel('\phi (rad)')
    ylabel('\omega (rad/s)')
    title('h(\phi,\omega) (V)') 
    axis([-pi pi -16*pi 16*pi])        
end
