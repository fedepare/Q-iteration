%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%
%%% SC4081 Knowledge Based Control Systems
%%%
%%% DC Motor Controlled with Fuzzy Reinforcement Learning Controller
%%%
%%% J. Lee (4089286), I. Matamoros (4510704), F. Paredes Vallés (4439953) and L. Valk (4095154)
%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

load results.mat

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

           [phi_alpha_vector, phi_omega_vector] = MF(xNowLimited, nAlphaTriang, nOmegaTriang, alpha_bounds, omega_bounds);

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

Width = 1.5;

%%% Plot the angle
subplot(4,1,1);
plot(time,X(1,:),'k','LineWidth',Width);
ylabel('\phi (rad)')
grid on
axis([0 0.5 -3.5 0.5])

%%% Plot the angular velocity
subplot(4,1,2);
plot(time,X(2,:),'k','LineWidth',Width);
ylabel('\omega (rad/s)')
grid on

%%% Plot the control signal
subplot(4,1,3);
plot(time,U(:),'k','LineWidth',Width);
ylabel('u (V)')
grid on

%%% Plot the reward signal
subplot(4,1,4);
plot(time,[reward(:)],'k','LineWidth',Width);
    ylabel('r (-)')
    xlabel('time (s)');  
    grid on
    axis([0 0.5 -60 10])
    
    
%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%
%%% Reconstruct fuzzy Q
%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
if(do_fuzzy_q_iteration)

    nPlotPoints = 100;
    alpha_values_plot = linspace(alpha_bounds(1),alpha_bounds(2),nPlotPoints);
    omega_values_plot = linspace(omega_bounds(1),omega_bounds(2),nPlotPoints);

    u_index = (nControlSteps+1)/2; % To see plot for u  0;
    Qplotvalues = zeros(nPlotPoints,nPlotPoints);
    
    for alpha_plot_index = 1:nPlotPoints
        for omega_plot_index = 1:nPlotPoints
            alpha_now = alpha_values_plot(alpha_plot_index);
            omega_now = omega_values_plot(omega_plot_index);
            xNow = [alpha_now;omega_now];
            
            [phi_alpha_vector, phi_omega_vector] = MF(xNow, nAlphaTriang, nOmegaTriang, alpha_bounds, omega_bounds);
            MuMatrix = min(  repmat(phi_omega_vector',[nAlphaTriang 1]) , repmat(phi_alpha_vector,[1 nOmegaTriang])  );
             
            Qplotvalues(alpha_plot_index,omega_plot_index) = sum(sum(MuMatrix.*Theta(:,:,u_index)));
        end
    end
end

figure(2)
[alpha_values_mesh,omega_values_mesh] = meshgrid(alpha_values_plot,omega_values_plot);
hFig = surf(alpha_values_mesh,omega_values_mesh,Qplotvalues);
colormap('gray')
colorbar
view(0,90)
xlabel('\phi (rad)')
ylabel('\omega (rad/s)')
title('Q(\phi,\omega,0) (-)')

%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%
%%% Reconstruct policy
%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
if(do_fuzzy_q_iteration)
    
    nPlotPoints = 50;
    alpha_values_plot = linspace(alpha_bounds(1),alpha_bounds(2),nPlotPoints);
    omega_values_plot = linspace(omega_bounds(1),omega_bounds(2),nPlotPoints);
    
    uplotvalues = zeros(nPlotPoints,nPlotPoints);
    
    for alpha_plot_index = 1:nPlotPoints
        for omega_plot_index = 1:nPlotPoints 
            for u_prime = 1:nControlSteps
                alpha_now = alpha_values_plot(alpha_plot_index);
                omega_now = omega_values_plot(omega_plot_index);
                xNow = [alpha_now;omega_now];

                [phi_alpha_vector, phi_omega_vector] = MF(xNow, nAlphaTriang, nOmegaTriang, alpha_bounds, omega_bounds);
                MuMatrix = min(  repmat(phi_omega_vector',[nAlphaTriang 1]) , repmat(phi_alpha_vector,[1 nOmegaTriang])  );
                SumdotMultiply(u_prime) = sum(sum(MuMatrix.*Theta(:,:,u_prime)));
            end

        [MuThetavalsMax,MuThetavalsMaxIndex]           = max(SumdotMultiply);  
        uplotvalues(alpha_plot_index,omega_plot_index) = u_values(MuThetavalsMaxIndex);
        end
    end
    
    figure(3)
    [alpha_values_mesh,omega_values_mesh] = meshgrid(alpha_values_plot,omega_values_plot);
    hFig = surf(alpha_values_mesh,omega_values_mesh,uplotvalues);  
    colormap('gray')
    colorbar
    view(0,90)
    xlabel('\phi (rad)')
    ylabel('\omega (rad/s)')
    title('h(\phi,\omega) (V)')
           
end