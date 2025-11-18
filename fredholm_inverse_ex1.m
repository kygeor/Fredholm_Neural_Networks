%% train_inverse_kernel.m
% Learn a kernel K_theta(x,y) from the forward data produced by run_fredholm_model.m
% Structure matches your original; only problem-specific parts are changed.

% Total number of training instances
total_instances = 10;
progress_file = 'training_progress_sin_sincos.mat';

% Resume logic
if exist(progress_file, 'file')
    loaded = load(progress_file, 'MSEs', 'models', 'lastInstance');
    MSEs = loaded.MSEs;
    models = loaded.models;
    start_instance = loaded.lastInstance + 1;
    fprintf('Resuming training from instance %d...\n', start_instance);
else
    MSEs = zeros(total_instances, 1);
    models = cell(total_instances, 1);
    start_instance = 1;
end

% Ensure forward variables exist
if ~exist('final_prediction', 'var') || ~exist('y', 'var') || ~exist('N', 'var') || ~exist('dy', 'var')
    error('Required variables (final_prediction, y, N, dy) not found in the workspace. Run run_fredholm_model.m first.');
end

%% Training Loop
for inst = start_instance:total_instances
    fprintf('\n--- Training Instance %d ---\n', inst);

    % 1) Data from forward run
    x_train  = y;                % grid (row/column handling inside helpers)
    f_target = final_prediction; % target f(x) on the same grid
    N_data   = N;                
    dx       = dy;

    fprintf('Size of f_target: %s\n', mat2str(size(f_target)));

    % 2) Additive term g(x) = sin(x)
    additive_part = @(value) sin(value);

    % 3) Kernel network (same small MLP as before)
    N_neurons = 20;
    net_kernel = feedforwardnet(N_neurons, 'trainlm');
    dummy_input = rand(2, 10);
    dummy_output = rand(1, 10);
    net_kernel = configure(net_kernel, dummy_input, dummy_output);
    net_kernel.layers{1}.transferFcn = 'tansig';
    net_kernel.layers{2}.transferFcn = 'purelin';

    % 4) Fredholm NN model 
    K_train = 15;  
    input_size  = numel(x_train);
    output_size = numel(x_train);

    grid_dictionary = struct();
    for i = 0:K_train
        field_name = sprintf('layer_%d', i);
        grid_dictionary.(field_name) = x_train;
    end

    % dummy kernel replaced during optimization
    dummy_kernel = @(in_val, out_val) 0;
    fred_model = FredholmNeuralNetwork(grid_dictionary, dummy_kernel, additive_part, additive_part, dx, K_train, input_size, output_size);

    % 5) Residual for lsqnonlin (with physics-style L2 regularization)
    model_error = @(wb) error_helper(net_kernel, wb, fred_model, x_train, f_target, dx, additive_part);

    % 6) Optimize with LM
    wb0 = getwb(net_kernel);
    options = optimoptions('lsqnonlin', ...
        'Display', 'iter-detailed', ...
        'Algorithm', 'levenberg-marquardt', ...
        'FunctionTolerance', 1e-8, ...
        'MaxIterations', 200);
    [wb_opt, resnorm, residual, exitflag, output] = lsqnonlin(model_error, wb0, [], [], options); 
    net_kernel = setwb(net_kernel, wb_opt);

    % 7) Evaluate learned model
    f_hat = compute_model_output(net_kernel, fred_model, x_train, dx, additive_part);
    mse_final = immse(f_hat, f_target);
    fprintf('Instance %d: Final MSE = %e\n', inst, mse_final);

    MSEs(inst) = mse_final;
    models{inst} = net_kernel;

    % 8) Save progress every 2 or at the end
    if mod(inst, 2) == 0 || inst == total_instances
        lastInstance = inst; 
        save(progress_file, 'MSEs', 'models', 'lastInstance');
        fprintf('Progress saved at instance %d.\n', inst);
    end
end

fprintf('Training complete. Results saved in %s.\n', progress_file);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% --- Local Functions  ---

function err = error_helper(net_kernel, wb, fred_model, x_train, f_target, dx, additive_part)
    % Update kernel network
    net_kernel = setwb(net_kernel, wb);

    % K_theta(x,y)
    new_kernel = @(in_val, out_val) evaluate_kernel(net_kernel, in_val, out_val);
    fred_model.kernel = new_kernel;

    % Forward through Fredholm NN (iterated operator)
    [f_FNN, ~] = fred_model.forward(x_train);

    % --- Data residual ---
    err_data = f_FNN - f_target;

    % --- Physics residual on the equation (your existing term) ---
    lambda_phys = 0e-6;                               % unchanged
    Kmat = evaluate_kernel(net_kernel, x_train, x_train);   % N×N
    integral_part = (Kmat * f_target) * dx;                 % N×1
    err_phys  = sqrt(lambda_phys) * (f_target(:) - integral_part - additive_part(x_train(:)));

    % --- kernel-size penalty: small penalty on ||K_theta|| ---
    % Row-wise L2 norms approximate ||K(x_i, ·)||_{L2(0,pi/2)}
    % multiply by sqrt(dx) for quadrature-consistent scaling.
    lambda_K = 1e-5;                                   
    row_l2 = sqrt(sum(Kmat.^2, 2)) * sqrt(dx);         % N×1
    err_K = sqrt(lambda_K) * row_l2;

    % Concatenate residuals
    err = [err_data(:); err_phys; err_K];
end


function f_hat = compute_model_output(net_kernel, fred_model, x_train, dx, additive_part) %#ok<INUSD>
    new_kernel = @(in_val, out_val) evaluate_kernel(net_kernel, in_val, out_val);
    fred_model.kernel = new_kernel;
    [f_hat, ~] = fred_model.forward(x_train);
end

function out = evaluate_kernel(net_kernel, in_val, out_val)
    % Ensure row/col
    in_val  = in_val(:).';  % 1×n  (y)
    out_val = out_val(:);   % m×1  (x)

    % Build (x,y) pairs -> m×n grids
    [X, Z] = ndgrid(out_val, in_val);  % X = x grid (m×n), Z = y grid (m×n)

    % NN input 2×(m*n)
    NN_input = [X(:)'; Z(:)'];

    % Evaluate and reshape to m×n
    out_vector = net_kernel(NN_input);
    out = reshape(out_vector, size(X));
end
