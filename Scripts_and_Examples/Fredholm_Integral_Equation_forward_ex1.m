%% run_fredholm_model.m
% Forward solve for: f(x) = sin(x) + ∫_0^{pi/2} sin(x) cos(y) f(y) dy

% --- Kernel & additive defining this specific FIE ---
% Kernel expects (in_value = y, out_value = x) per the class' convention.
kernel   = @(in_value, out_value) 0.5 * sin(out_value) .* cos(in_value);  % m×n by implicit expansion
additive = @(value) sin(value);

% --- Discretization / network params (unchanged structure) ---
K = 10;             % Number of iterations/layers (excluding layer_0)
N = 300;            % Grid points
input_size  = N;
output_size = N;

% Grid on [0, pi/2]
ymin = 0; ymax = pi/2;
y = linspace(ymin, ymax, N);

% Build grid_dictionary with same grid per layer
grid_dictionary = struct();
for i = 0:K
    field_name = sprintf('layer_%d', i);
    grid_dictionary.(field_name) = y;
end

% Integration step
dy = (y(end) - y(1)) / (N - 1);

% Predict array 
predict_array = y;

% Instantiate and run
model = FredholmNeuralNetwork(grid_dictionary, kernel, additive, additive, dy, K, input_size, output_size);
[final_prediction, layer_outputs] = model.forward(predict_array);

% Shapes per layer
for i = 1:length(layer_outputs)
    layer_output = layer_outputs{i};
    fprintf('Layer %d output shape: %s\n', i-1, mat2str(size(layer_output)));
end

% Plot
figure;
plot(y, final_prediction, 'LineWidth', 2);
xlabel('x'); ylabel('f(x)');
title('Forward solution: f(x) for K(x,y)=sin(x)cos(y), g(x)=sin(x)');
grid on;
