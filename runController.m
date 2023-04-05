function [time, nominal_trajectory, linearized_trajectory, reactive_trajectory, anticipatory_trajectory, reactive_dosage, anticipatory_dosage] = runController(bsa_arg, num_cycles_arg, dosage, anc_measurements)
    %% 2. Trajectory with Jost et al. model
    %% 2.1 Intialization of constants
    bsa = bsa_arg;
    theta = [31.2;12.72;0.019;9.9216;0.219*(bsa^1.16);2.06;0.146;0.103;0.866;2.3765];
    % initial values (from Jost et al.)
    x0 = [0;...
        0;...
        0;...
        theta(6)*theta(10)/theta(7);...
        theta(6)*theta(10)/theta(7);...
        theta(6)*theta(10)/theta(7);...
        theta(6)*theta(10)/theta(7);...
        theta(6)];
    
    num_cycles = num_cycles_arg;
    
    %% 2.2 Calculation of reference trajectory for 34 cycles
    %x_ref is the states (including neutrophil count)
    x_ref = [];
    t_ref = [];
    u_ref = [];
    step_size_ref = 0.01; % very fine step size
    num_t_ref = (1/step_size_ref)*21+1; % number of timepoints to evaluate ODE45
    tspan = linspace(0,21,num_t_ref); %21 days
    
    dosage = cell2mat(dosage);
    historical_dosages = dosage*bsa; % 6-MP dosage (mg 6-MP/m^2 body surface area)
    anc_measurements = cell2mat(anc_measurements);
    x0_i = x0;
    %1 cycle is 21 days
    for i=1:num_cycles
        
        if i <= length(historical_dosages)
            u_i = historical_dosages(i);
        end

        if i <= length(anc_measurements)
            x0_i(8) = anc_measurements(i);
        end

        u_ref = [u_ref [u_i*(ones((1/step_size_ref)*14,1));zeros((1/step_size_ref)*(21-14)+1,1)]];
        [t_i,x_i] = ode45(@(t,x)jost(t,x,u_i,theta),tspan,x0_i);
        t_i = t_i + 21*(i-1); % shift the time index to the current cycle
        % check lower bound
        
        if x_i(end,8) < 1
            u_i = 0.8*u_i;
        elseif x_i(end,8) > 2 % check upper bound
            u_i = 1.2*u_i;
        end %readjust the dosage
        x_ref = [x_ref x_i];
        t_ref = [t_ref t_i];
        x0_i = x_i(end,:); %start the new cycle at the last state of the previous cycle
    end
    
    clear i t_i x_i x0_i u_i tspan;
    
    % flatten to single nx8 matrix
    x_ref_flattened = [];
    t_ref_flattened = [];
    u_ref_flattened = [];
    for i=1:num_cycles
        x_ref_flattened = [x_ref_flattened; x_ref(:,8*(i-1)+1:8*i)];
        t_ref_flattened = [t_ref_flattened; t_ref(:,i)];
        u_ref_flattened = [u_ref_flattened; u_ref(:,i)];
    end
    clear i;

    nominal_trajectory = x_ref_flattened(:,8);
    
    %% 2.6 Linearization
    
    bsa = bsa_arg; % trajectory near original bsa
    theta = [31.2;12.72;0.019;9.9216;0.219*(bsa^1.16);2.06;0.146;0.103;0.866;2.3765];
    x_lin = [];
    t_lin = [];
    u_lin = [];
    step_size_lin = 0.01; % coarse step size; NOTE: because of forward euler's small stability region, the largest I could get this is 0.05 with a different bsa
    num_t_lin = (1/step_size_lin)*21+1; % number of timepoints to evaluate ODE45
    tspan = linspace(0,21,num_t_lin);
    
    
    t_lin_j_start = 0; % current time at end of last cycle
    x_t = x0; % initial "guess"
    % u_i = dosage*bsa;
    
    
    dx_t = zeros(8,1); % keep track of delta x
    %du_t = 0; % keep track of delta u
    
    dx_t_arr = [];
    
    for i=1:num_cycles
        
        if i <= length(historical_dosages)
            u_i = historical_dosages(i);
        end

        u_lin = [u_lin u_i];
        u_star = u_ref(end,i);
        du_t = u_i - u_star;
    
        x_ref_i = x_ref(:,8*(i-1)+1:8*i);
        x_lin_j = [];
        t_lin_j = [];
        
        for j=1:length(tspan)
            x_star = transpose(x_ref_i(round((j-1)*step_size_lin/step_size_ref)+1,:));
            %round((j-1)*step_size_lin/step_size_ref)+1
            x_t = x_star+dx_t;
            x_lin_j = [x_lin_j; transpose(x_t)];
    
            [f, dfdx, dfdu, dgdx] = jost_fwd_euler(tspan(j),x_star,u_star,theta,step_size_lin);
            dx_t = dfdx*dx_t+dfdu*du_t;
            if tspan(j) >= 14
                % u_i and u_ref(i) are both zero after 14 days
                du_t = 0;
            end
            t_lin_j = [t_lin_j; t_lin_j_start+tspan(j)];
        end
        t_lin = [t_lin t_lin_j];
        x_lin = [x_lin x_lin_j];
    
        % check lower bound
        if x_t(8) < 1
            u_i = 0.8*u_i;
        elseif x_t(8) > 2 % check upper bound
            u_i = 1.2*u_i;
        end
        t_lin_j_start = t_lin_j_start + 21; % shift the time index to the current cycle
    end
    
    t_lin_flattened = [];
    x_lin_flattened = [];
    for i=1:num_cycles
        x_lin_flattened = [x_lin_flattened; x_lin(:,8*(i-1)+1:8*i)];
        t_lin_flattened = [t_lin_flattened; t_lin(:,i)];
    end

    linearized_trajectory = x_lin_flattened(:,8);
    
    clear i j x_lin_j x0_i u_i tspan;
    
    %% 2.7.1 Trajectory with Noise; Setup
    
    num_cycles = num_cycles_arg; 
    
    step_size_noisy = step_size_ref;
    num_t_noisy = (1/step_size_noisy)*21+1; % number of timepoints to evaluate ODE45
    tspan = linspace(0,21,num_t_noisy);
    
    var_w = 0.01; % TODO: change this internal system noise (currently arbitrary)
    var_v = 0.774341747; % CHECK if documented measurement variance
    
    %% 2.7.2 Trajectory with Noise; Reactive Controller
    tic
    
    u_i = dosage*bsa;
    x0_i = x0;
    
    y = [];
    
    x_noisy_r = [];
    t_noisy_r = [];
    u_noisy_r = [];
    
    
    for i=1:num_cycles

        if i <= length(historical_dosages)
            u_i = historical_dosages(i);
        end

        if i <= length(anc_measurements)
            x0_i(8) = anc_measurements(i);
        end
        
        u_i_all = [transpose(repelem(u_i,(1/step_size_noisy)*14));transpose(repelem(0,(1/step_size_noisy)*(21-14)+1))];
        u_noisy_r = [u_noisy_r u_i_all];
        [t_i,x_i] = ode45(@(t,x)jost_noisy(t,x,u_i,theta,step_size_noisy,var_w),tspan,x0_i);
        t_i = t_i + 20*(i-1); % shift the time index to the current cycle
        
        % make a "reading"
        % ground truth (at the end of the cycle)
        y_true = x_ref(end,8*i);
        % sample a reading
        y_i = normrnd(y_true,var_v);
        y = [y y_i];
    
        % check lower bound
        if y_i < 1
            u_i = 0.8*u_i;
        elseif y_i > 2 % check upper bound
            u_i = 1.2*u_i;
        end
        x_noisy_r = [x_noisy_r x_i];
        t_noisy_r = [t_noisy_r t_i];
        
        x0_i = x_i(end,:);
    end
    
    clear i t_i x_i x0_i u_i tspan;
    
    % flatten to single nx8 matrix
    x_noisy_r_flattened = [];
    t_noisy_r_flattened = [];
    u_noisy_r_flattened = [];
    for i=1:num_cycles
        x_noisy_r_flattened = [x_noisy_r_flattened; x_noisy_r(:,8*(i-1)+1:8*i)];
        t_noisy_r_flattened = [t_noisy_r_flattened; t_noisy_r(:,i)];
        u_noisy_r_flattened = [u_noisy_r_flattened; u_noisy_r(:,i)];
    end
    clear i;
    
    toc
    
    reactive_trajectory = x_noisy_r_flattened(:,8);
    
    %% 2.7.3 Trajectory with Noise; KF Controller 
    tic
    
    % Simu_symlation Length
    t_s = step_size_noisy;
    t_end = 21*num_cycles;
    N = t_end/t_s;
    time = [0:t_s:t_end];
    
    % Dimension
    % n: dimension of state vector x
    n = 8;
    % m: dimension of observation vector y
    m = 1;
    
    % Initialize motion function
    syms x1_sym x2_sym x3_sym x4_sym x5_sym x6_sym x7_sym x8_sym u_sym
    
    f = [-theta(1)                0 0 0 0 0 0 0; ...
        theta(1) -theta(2)          0 0 0 0 0 0; ...
        0 theta(3)*theta(4) -theta(5) 0 0 0 0 0; ...
        0 0 0 -theta(7)                 0 0 0 0; ...
        0 0 0 theta(7) -theta(7)          0 0 0; ...
        0 0 0 0 theta(7) -theta(7)          0 0; ...
        0 0 0 0 0 theta(7) -theta(7)          0; ...
        0 0 0 0 0 0 theta(7) -theta(10)]*[x1_sym;x2_sym;x3_sym;x4_sym;x5_sym;x6_sym;x7_sym;x8_sym] + ...
        [0.22;0;0;0;0;0;0;0]*u_sym + ...
        [0;...
        0;...
        0;...
        theta(7)*((theta(6)/x8_sym)^theta(9))*x4_sym-theta(7)*theta(8)*((theta(6)/x8_sym)^theta(9))*x3_sym*x4_sym;...
        0;...
        0;...
        0;...
        0];
    
    f = [x1_sym;x2_sym;x3_sym;x4_sym;x5_sym;x6_sym;x7_sym;x8_sym] + t_s * f;
    
    dfdx = jacobian(f,[x1_sym;x2_sym;x3_sym;x4_sym;x5_sym;x6_sym;x7_sym;x8_sym]);
    dfdu = jacobian(f,u_sym);
    
    % Initialize observation function
    g = [x8_sym];
    dgdx = jacobian(g,[x1_sym;x2_sym;x3_sym;x4_sym;x5_sym;x6_sym;x7_sym;x8_sym]);
    
    % Initialize disturbance xi and covariance Q
    xi = zeros([n 1 N]);
    Q = zeros([n n N]);
    Q_const = var_w * eye(n);
    for iter = 1:N
        Q(:,:,iter) = Q_const;
        xi(:,:,iter) = Q(:,:,iter) * randn([n,1]);
    end
    clear Q_const
    
    % Initialize measurement noise n and covariance R
    nn = zeros([m 1 N]);
    R = zeros([m m N]);
    R_const = var_v * eye(m);
    for iter = 1:N
        R(:,:,iter) = R_const;
        nn(:,:,iter) = R(:,:,iter) * randn([m,1]);
    end
    clear R_const
    
    % Initialize nominal trajectory and control to track
    % nominal control is set as linear for test purposes
    A_nominal = zeros([n n N]);
    b_nominal = zeros([n N]);
    
    % convert and reshape for Sam's code
    u_nominal = transpose(u_ref_flattened(1:N,1));
    
    
    x_nominal = zeros([n 1 N]);
    % Simulate System for the nominals
    x_o = x0;
    for k = 1:N
        % Compute linearized c2d-ed state matrices, we use this to simulate the
        % system because we do not know the discrete time nonlinear equation
        % for the inverse pendulum problem
        if k == 1
            x_tmp = x_o;
        else
            x_tmp = x_nominal(:,:,k-1);
        end
        [A_actual,b_actual,~] = c2d_jost(x1_sym,x2_sym,x3_sym,x4_sym,x5_sym,x6_sym,x7_sym,x8_sym,u_sym,dfdx,dfdu,dgdx,x_tmp,u_nominal(:,k),t_s);
        A_nominal(:,:,k) = A_actual;
        b_nominal(:,k) = b_actual;
        % Time step
        x_nominal(:,:,k) = double(subs(f,{x1_sym,x2_sym,x3_sym,x4_sym,x5_sym,x6_sym,x7_sym,x8_sym,u_sym},{x_tmp(1),x_tmp(2),x_tmp(3),x_tmp(4),x_tmp(5),x_tmp(6),x_tmp(7),x_tmp(8),u_nominal(:,k)}));
    end
    clear k x_tmp A_actual b_actual
    
    
    % Initialize Cost Parameters
    W = zeros([n n N+1]);
    rho = zeros([n 1 N+1]);
    lambda = zeros([1 N]);
    W_const = 100 * eye(n);
    lambda_const = 0.1;
    for iter = 1:N+1
        W(:,:,iter) = W_const;
        if iter == 1
            rho(:,:,iter) = x_o;
        else
            rho(:,:,iter) = x_nominal(:,:,iter-1);
        end
    end
    for iter = 1:N
        lambda(:,iter) = lambda_const;
    end
    clear W_const lambda_const
    
    % Note: Iteration variables does not correspond to the notes due to matlab
    % indexing start with 1 instead of 0
    
    x_o = x0;
    x_hat_o = x0; 
    sigma_o = eye(n);
    
    % Simulate System with control
    is_controlled = 1;
    [x_c,y_c,x_hat_p_c,u_c] = simulate_jost(N,n,m,t_s,x1_sym,x2_sym,x3_sym,x4_sym,x5_sym,x6_sym,x7_sym,x8_sym,u_sym,f,dfdx,dfdu,dgdx,xi,Q,nn,R,W,rho,lambda,x_o,x_hat_o,sigma_o,A_nominal,b_nominal,x_nominal,u_nominal,is_controlled);
    
    
    % Process Result
    x1_c_plot = [x_o(1,1);reshape(x_c(1,1,:),[N 1])];
    x2_c_plot = [x_o(2,1);reshape(x_c(2,1,:),[N 1])];
    x3_c_plot = [x_o(3,1);reshape(x_c(3,1,:),[N 1])];
    x4_c_plot = [x_o(4,1);reshape(x_c(4,1,:),[N 1])];
    x5_c_plot = [x_o(5,1);reshape(x_c(5,1,:),[N 1])];
    x6_c_plot = [x_o(6,1);reshape(x_c(6,1,:),[N 1])];
    x7_c_plot = [x_o(7,1);reshape(x_c(7,1,:),[N 1])];
    x8_c_plot = [x_o(8,1);reshape(x_c(8,1,:),[N 1])];
    y1_c_plot = [x_o(1,1);reshape(y_c(1,1,:),[N 1])];
    x1_hat_c_plot = [x_hat_o(1,1);reshape(x_hat_p_c(1,1,:),[N 1])];
    x2_hat_c_plot = [x_hat_o(2,1);reshape(x_hat_p_c(2,1,:),[N 1])];
    x3_hat_c_plot = [x_hat_o(3,1);reshape(x_hat_p_c(3,1,:),[N 1])];
    x4_hat_c_plot = [x_hat_o(4,1);reshape(x_hat_p_c(4,1,:),[N 1])];
    x5_hat_c_plot = [x_hat_o(5,1);reshape(x_hat_p_c(5,1,:),[N 1])];
    x6_hat_c_plot = [x_hat_o(6,1);reshape(x_hat_p_c(6,1,:),[N 1])];
    x7_hat_c_plot = [x_hat_o(7,1);reshape(x_hat_p_c(7,1,:),[N 1])];
    x8_hat_c_plot = [x_hat_o(8,1);reshape(x_hat_p_c(8,1,:),[N 1])];
    
    anticipatory_trajectory = x8_c_plot;

    rho8_plot = reshape(rho(8,1,:),[N+1 1]); %anticipatory
    
    toc
    %time = t_ref_flattened;
    %anticipatory_dosage = u_c.';
    reactive_dosage = u_noisy_r_flattened / bsa;

    tic

    %% https://github.com/YuYing-Liang/Control-Strategies-for-Leukemia-Treatment/blob/main/kalman-filter-lqr/model/nominal_trajectory.m#L705

    %% 2.8.1 Control Input Smoothing - Cyclic Intervals

    N = length(u_c);
    u_smoothed_flattened = zeros(1, N);
    u_smoothed = [];
    t = time(1:N);
    
    i=1;
    start_i = 1;
    day = 14;
    t_i = t(i);
    sum = 0;
    n_points = 0;
    
    while i <= N
        sum = sum + u_c(i);
        if t(i) == day
            u_i = sum/n_points;
            u_smoothed_flattened(start_i:i) = u_i;
            idx_7_days_later = find(t == day+6);
            u_smoothed_flattened(i:idx_7_days_later) = 0;
            u_smoothed = [u_smoothed u_i];
    
            i = idx_7_days_later;
            n_points = 0;
            sum = 0;
            start_i = i+1;
            day = day + 20;
            if day > N
                day = N;
            end
        else
            n_points = n_points + 1;
        end
        i = i + 1;
    end
    
    if n_points > 0
        u_smoothed_flattened(start_i:i-1) = sum/n_points;
        u_smoothed = [u_smoothed u_i];
    end
    
    clear t start_i i day idx_7_days_later sum u_i
    
    anticipatory_dosage = u_smoothed_flattened.' / bsa;
    
    %% 2.8.2 Model's Response with Smoothed Inputs - Cyclic Intervals
    
    x_smoothed = [];
    t_smoothed = [];
    step_size_smoothed = 0.01; % very fine step size
    num_t_smoothed = (1/step_size_smoothed)*21+1; % number of timepoints to evaluate ODE45
    tspan = linspace(0,21,num_t_smoothed);
    x0_i = x0;
        
    for i=1:num_cycles
        [t_i,x_i] = ode45(@(t,x)jost(t,x,u_smoothed(i),theta),tspan,x0_i);
        t_i = t_i + 21*(i-1); % shift the time index to the current cycle
    
        x_smoothed = [x_smoothed x_i];
        t_smoothed = [t_smoothed t_i];
        x0_i = x_i(end,:);
    end
    
    clear i t_i x_i x0_i u_i tspan;
    
    % flatten to single nx8 matrix
    x_smoothed_flattened = [];
    t_smoothed_flattened = [];
    for i=1:num_cycles
        x_smoothed_flattened = [x_smoothed_flattened; x_smoothed(:,8*(i-1)+1:8*i)];
        t_smoothed_flattened = [t_smoothed_flattened; t_smoothed(:,i)];
    end
    clear i;

    time = t_ref_flattened;
    toc
    
end
