%% Golf Game Reinforcement Learning - SARSA(λ) with Eligibility Traces
% Complete implementation of SARSA(λ) with proper eligibility traces
% Course: AE4350 Bio-inspired Intelligence

clear; clc; close all;

%% Algorithm Configuration
fprintf('=== SARSA(λ) with Eligibility Traces ===\n');

% Eligibility Trace Parameters
lambda = 0.4;              % Eligibility trace decay (0 = one-step SARSA, 1 = Monte Carlo)
trace_type = 'replacing';  % Options: 'accumulating', 'replacing', 'dutch'

% Learning Parameters
alpha = 0.025;     
gamma = 0.95;              % Discount factor
epsilon_start = 1.0;       % Initial exploration rate
epsilon_min = 0.01;        % Minimum exploration rate
epsilon_decay = 0.9995;    % Epsilon decay per episode

% Training Parameters
num_episodes = 20000;      % Total episodes
max_steps_per_episode = 1000;

% Progress reporting
report_interval = 1000;

fprintf('Lambda: %.2f | Trace type: %s\n', lambda, trace_type);
fprintf('Learning rate: %.3f | Discount: %.2f\n', alpha, gamma);

%% Game Environment Setup
% Define continuous field dimensions
field_width = 10;  % Width 
field_height = 8;  % Height 

% Start and goal positions (continuous)
goal_pos = [1.0, 7.0];   % Upper-left area
start_pos = [9.0, 1.0];  % Lower-right area
goal_radius = 0.2;       % Radius for reaching the goal

% Water hazards 
obstacles = [
    3.0, 1.6, 6.0, 2.4;   
    3.0, 0, 6.0, 0.8;   
    5.0, 5.0, 6.5, 6.4
    7.5, 6.0, 10.0, 8.0
    3.5, 7.2, 7.5, 8.0];             

% Walls 
walls = struct();
walls.segments = [
   
    2.0, 2.5, 10.0, 2.5;   
    6.5, 4.0, 6.5, 5.0;   
    6.5, 2.5, 6.5, 3.5;   
    5, 3.2, 5, 4.2;    
    3.5, 6, 3.5, 8;    
    0.0, 5.0, 6.5, 5.0;    
];

%% Action Space Configuration
num_directions = 36;                   % Number of directions
num_force_levels = 14;                 % Number of force levels
force_values =[0.1, 0.2, 0.3, 0.4, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0, 7.0, 9.0];  
num_actions = num_directions * num_force_levels;

fprintf('Actions: %d directions × %d forces = %d total\n', num_directions, num_force_levels, num_actions);

%% Tile Coding Setup
num_tilings = 4;
tiles_per_dim = 20;
tile_width = field_width / tiles_per_dim;
tile_height = field_height / tiles_per_dim;

% Create random offsets for tilings
tile_offsets = rand(num_tilings, 2) .* [tile_width, tile_height];

% Total features
total_tiles = num_tilings * tiles_per_dim * tiles_per_dim;
fprintf('State features: %d tilings × %d² tiles = %d total\n\n', num_tilings, tiles_per_dim, total_tiles);

% Adjusted learning rate
% alpha = alpha_base / num_tilings;

%% Initialize Learning Structures
% Weights for linear function approximation
weights = zeros(total_tiles, num_actions);

% Eligibility traces for each state-action pair
eligibility_traces = sparse(total_tiles, num_actions);

% Performance tracking
episode_steps = zeros(num_episodes, 1);
episode_rewards = zeros(num_episodes, 1);
success_count = 0;

%% Training Loop
fprintf('Starting SARSA(λ) training...\n');
fprintf('Episode    | Avg Steps | Success Rate | Epsilon\n');
fprintf('-----------|-----------|--------------|--------\n');

tic;
epsilon = epsilon_start;

for episode = 1:num_episodes
    % Reset eligibility traces at start of each episode
    eligibility_traces = sparse(total_tiles, num_actions);
    
    % Initialize episode
    ball_pos = start_pos;
    
    % Get initial state features
    features = get_tile_features(ball_pos, num_tilings, tiles_per_dim, ...
                                tile_width, tile_height, tile_offsets, field_width, field_height);
    
    % Choose initial action using ε-greedy
    if rand() < epsilon
        action = randi(num_actions);
    else
        Q_values = weights' * features;
        [~, action] = max(Q_values);
    end
    
    % Episode variables
    done = false;
    step_count = 0;
    total_reward = 0;
    
    while ~done && step_count < max_steps_per_episode
        step_count = step_count + 1;
        
        % Take action and observe result
        [new_pos, reward, done] = take_continuous_action_sarsa(ball_pos, action, ...
            num_directions, num_force_levels, force_values, field_width, field_height, ...
            goal_pos, goal_radius, obstacles, walls);
        
        total_reward = total_reward + reward;
        
        % Get new state features
        new_features = get_tile_features(new_pos, num_tilings, tiles_per_dim, ...
                                       tile_width, tile_height, tile_offsets, field_width, field_height);
        
        % Choose next action (SARSA: need actual next action)
        if ~done  % Only choose next action if not terminal
            if rand() < epsilon
                next_action = randi(num_actions);
            else
                Q_values_next = weights' * new_features;
                [~, next_action] = max(Q_values_next);
            end
        else
            next_action = 1;  % Dummy value, won't be used
        end
        
        % Calculate TD error
        Q_current = weights(:, action)' * features;
        
        if done
            if norm(new_pos - goal_pos) <= goal_radius  % Reached goal
                delta = reward - Q_current;
                success_count = success_count + 1;
            else  % Hit water or timeout
                delta = reward - Q_current;
            end
        else
            Q_next = weights(:, next_action)' * new_features;
            delta = reward + gamma * Q_next - Q_current;
        end
        
        %Decay traces
        eligibility_traces = gamma * lambda * eligibility_traces;
        
        % Update eligibility traces based on trace type
        switch trace_type
            case 'accumulating'
                
                eligibility_traces(:, action) = eligibility_traces(:, action) + features;
                
            case 'replacing'
                
                eligibility_traces(:, action) = features;
                
            case 'dutch'
             
                eligibility_traces(:, action) = eligibility_traces(:, action) + ...
                                               features - alpha * features .* (features' * eligibility_traces(:, action));
        end
        
   
        
        % Update only non-zero traces
       [i, j, v] = find(eligibility_traces);  % Get non-zero elements
       for idx = 1:length(i)
           weights(i(idx), j(idx)) = weights(i(idx), j(idx)) + alpha * delta * v(idx);
       end
        
        % Move to next state
        ball_pos = new_pos;
        features = new_features;
        action = next_action;
    end
    
    % Store episode results
    episode_steps(episode) = step_count;
    episode_rewards(episode) = total_reward;
    
    % Decay exploration
    epsilon = max(epsilon_min, epsilon * epsilon_decay);
    
    % Progress report
    if mod(episode, report_interval) == 0
        avg_steps = mean(episode_steps(max(1, episode-999):episode));
        recent_success = success_count / episode;
        fprintf('%10d | %9.1f | %11.1f%% | %.4f\n', ...
                episode, avg_steps, recent_success*100, epsilon);
    end
end

training_time = toc;

%% Final Statistics
fprintf('\n=== Training Complete ===\n');
fprintf('Total time: %.2f seconds (%.3f ms/episode)\n', training_time, training_time/num_episodes*1000);
fprintf('Total successes: %d/%d (%.1f%%)\n', success_count, num_episodes, success_count/num_episodes*100);

% Save weights
save('sarsa_lambda_weights.mat', 'weights', 'lambda', 'trace_type', 'tile_offsets');
fprintf('\nWeights saved to sarsa_lambda_weights.mat\n');

%% Test Learned Policy
fprintf('\n=== Testing Learned Policy ===\n');
test_episodes = 1;
test_successes = 0;
test_total_steps = 0;

for test = 1:test_episodes
    test_pos = start_pos;
      test_done = false;
    test_steps = 0;
    test_path = test_pos;
    
    while ~test_done && test_steps < max_steps_per_episode
        test_steps = test_steps + 1;
        
        % Get features
        test_features = get_tile_features(test_pos, num_tilings, tiles_per_dim, ...
                                        tile_width, tile_height, tile_offsets, field_width, field_height);
        
        % Greedy action selection
        Q_values = weights' * test_features;
        [~, test_action] = max(Q_values);
        
        % Take action
        [new_test_pos, test_reward, test_done] = take_continuous_action_sarsa(test_pos, test_action, ...
            num_directions, num_force_levels, force_values, field_width, field_height, ...
            goal_pos, goal_radius, obstacles, walls);
        
        test_pos = new_test_pos;
        test_path = [test_path; test_pos];
        
        if norm(test_pos - goal_pos) <= goal_radius
            test_successes = test_successes + 1;
        end
    end
    
    test_total_steps = test_total_steps + test_steps;
    fprintf('Test %2d: %s in %d steps\n', test, ...
            iif(test_reward > 0, 'SUCCESS', 'FAILED'), test_steps);
end

fprintf('\nTest results: %d/%d successful (%.0f%%), avg steps: %.1f\n', ...
        test_successes, test_episodes, test_successes/test_episodes*100, ...
        test_total_steps/test_episodes);

%% Visualization
visualize_sarsa_results(field_width, field_height, test_path, obstacles, goal_pos, ...
                       goal_radius, start_pos, walls,tiles_per_dim,tile_width, tile_height, episode_steps, episode_rewards, ...
                       lambda, trace_type);

%% Helper Functions

function features = get_tile_features(pos, num_tilings, tiles_per_dim, ...
                                    tile_width, tile_height, tile_offsets, ...
                                    field_width, field_height)
    
    % Convert continuous position to tile features
    total_tiles = num_tilings * tiles_per_dim * tiles_per_dim;
    features = zeros(total_tiles, 1);
    
    % Ensure position is row vector
    pos = pos(:)';
    
    % Clamp to field bounds
    pos(1) = max(0, min(field_width, pos(1)));
    pos(2) = max(0, min(field_height, pos(2)));
    
    for tiling = 1:num_tilings
        % Apply offset for this tiling
        offset_pos = pos - tile_offsets(tiling, :);
        
        % Wrap around
        offset_pos(1) = mod(offset_pos(1), field_width);
        offset_pos(2) = mod(offset_pos(2), field_height);
        
        % Get tile indices
        tile_x = floor(offset_pos(1) / tile_width);
        tile_y = floor(offset_pos(2) / tile_height);
        
        % Ensure within bounds
        tile_x = max(0, min(tiles_per_dim - 1, tile_x));
        tile_y = max(0, min(tiles_per_dim - 1, tile_y));
        
        % Convert to linear index
        tile_index = tile_y * tiles_per_dim + tile_x + 1;
        feature_index = (tiling - 1) * tiles_per_dim * tiles_per_dim + tile_index;
        
        features(feature_index) = 1;
    end
end

function [new_pos, reward, done] = take_continuous_action_sarsa(ball_pos, action_idx, ...
    num_directions, num_force_levels, force_values, field_width, field_height, ...
    goal_pos, goal_radius, obstacles, walls)
    
    % Decode action
    [direction, force] = decode_action(action_idx, num_directions, num_force_levels, force_values);
    
    % Calculate angle 
    angle = (direction - 1) * (2 * pi / num_directions);
    
    % Calculate movement
    dx = force * cos(angle);
    dy = force * sin(angle);
    
    % New position
    intended_pos = ball_pos + [dx, dy];
    
    % Clamp to field boundaries
    intended_pos(1) = max(0.2, min(intended_pos(1), field_width - 0.2));
    intended_pos(2) = max(0.2, min(intended_pos(2), field_height - 0.2));

    % Check wall collision
    if check_wall_collision(ball_pos, intended_pos, walls) || crosses_water_hazard(ball_pos, intended_pos, obstacles)
        new_pos = ball_pos;  % No movement if collision
    else
        new_pos = intended_pos;
    end
    
    % Calculate reward and check if done
    reward = get_continuous_reward(ball_pos, new_pos, obstacles, goal_pos, goal_radius);
    done = check_continuous_done(new_pos, obstacles, goal_pos, goal_radius);
end

function [direction, force] = decode_action(action_idx, num_directions, num_force_levels, force_values)
    % Decode action index into direction and force
    direction = mod(action_idx - 1, num_directions) + 1;
    force_idx = floor((action_idx - 1) / num_directions) + 1;
    force = force_values(force_idx);
end

function reward = get_continuous_reward(pos, new_pos, obstacles, goal_pos, goal_radius)
    % Calculate reward in continuous space 
    
    dist_to_goal = norm(new_pos - goal_pos);
    shaping_radius = 1.5;  
    
    % Default step penalty
    reward = -6;

   
   % Check if landed in water
   for i = 1:size(obstacles, 1)
       if new_pos(1) >= obstacles(i, 1) && new_pos(1) <= obstacles(i, 3) && ...
          new_pos(2) >= obstacles(i, 2) && new_pos(2) <= obstacles(i, 4)
          reward = -20;
          return;
       end
   end

    % Check if stayed in same position (hit the wall or crossed the water)
    if norm(pos - new_pos) < 1e-6
        reward = -10;
        return;
    end
    
    % Check if reached goal
    if dist_to_goal <= goal_radius
        reward = 50;
        return;
    end


     % Distance-based shaping 
    if dist_to_goal <= shaping_radius
        % Linearly interpolate bonus from +0 to +5
        bonus = 5 * (shaping_radius - dist_to_goal) / shaping_radius;
        reward = reward + bonus;  % Add to step penalty
    end

end

function done = check_continuous_done(pos, obstacles, goal_pos, goal_radius)
    % Check if episode should end
    
    % Goal reached
    if norm(pos - goal_pos) <= goal_radius
        done = true;
        return;
    end
    
    % In water hazard
    for i = 1:size(obstacles, 1)
        if pos(1) >= obstacles(i, 1) && pos(1) <= obstacles(i, 3) && ...
           pos(2) >= obstacles(i, 2) && pos(2) <= obstacles(i, 4)
            done = true;
            return;
        end
    end
    
    done = false;
end

function collision = check_wall_collision(pos1, pos2, walls)
    % Check if movement collides with any wall
    collision = false;
    
    if ~isfield(walls, 'segments') || isempty(walls.segments)
        return;
    end
    
    for i = 1:size(walls.segments, 1)
        wall_start = walls.segments(i, 1:2);
        wall_end = walls.segments(i, 3:4);
        
        if line_segments_intersect(pos1, pos2, wall_start, wall_end)
            collision = true;
            return;
        end
    end
end

function crossed = crosses_water_hazard(pos1, pos2, obstacles)
    % Check if path crosses water
    crossed = false;
    
    for i = 1:size(obstacles, 1)
        % Check intersection with obstacle edges
        corners = [
            obstacles(i, 1), obstacles(i, 2);
            obstacles(i, 3), obstacles(i, 2);
            obstacles(i, 3), obstacles(i, 4);
            obstacles(i, 1), obstacles(i, 4)
        ];
        
        for j = 1:4
            p1 = corners(j, :);
            p2 = corners(mod(j, 4) + 1, :);
            
            if line_segments_intersect(pos1, pos2, p1, p2)
                crossed = true;
                return;
            end
        end
    end
end

function intersect = line_segments_intersect(p1, p2, p3, p4)
    % Check if line segments intersect
    d1 = p2 - p1;
    d2 = p4 - p3;
    
    det = d1(1) * d2(2) - d1(2) * d2(1);
    
    if abs(det) < 1e-10
        intersect = false;
        return;
    end
    
    dp = p1 - p3;
    t = (d2(1) * dp(2) - d2(2) * dp(1)) / det;
    u = (d1(1) * dp(2) - d1(2) * dp(1)) / det;
    
    intersect = (t >= 0 && t <= 1 && u >= 0 && u <= 1);
end

function result = iif(condition, true_val, false_val)
    if condition
        result = true_val;
    else
        result = false_val;
    end
end

function visualize_sarsa_results(field_width, field_height, path, obstacles, goal_pos, goal_radius, ...
                                    start_pos, walls, tiles_per_dim, ...
                                    tile_width, tile_height, episode_steps, episode_rewards,lambda,trace_type)
    
    figure('Position', [50, 50, 1400, 800]);
    
    % Plot 1: Golf course with path
    subplot(2, 2, [1, 3]);
    
    % Draw field background
    rectangle('Position', [0, 0, field_width, field_height], ...
              'FaceColor', [0.2 0.8 0.2], 'EdgeColor', 'k', 'LineWidth', 2);
    hold on;
    
    % Add axis labels to clarify coordinate system
    text(5, -0.5, 'X axis →', 'HorizontalAlignment', 'center', 'FontSize', 10);
    text(-0.5, 4, 'Y axis ↑', 'HorizontalAlignment', 'center', 'FontSize', 10, 'Rotation', 90);
    
    % Draw water hazards
    for i = 1:size(obstacles, 1)
        rectangle('Position', [obstacles(i, 1), obstacles(i, 2), ...
                              obstacles(i, 3) - obstacles(i, 1), ...
                              obstacles(i, 4) - obstacles(i, 2)], ...
                  'FaceColor', [0.1 0.4 0.8], 'EdgeColor', 'k');
    end
    
    % Draw walls with thicker lines and end markers
    for i = 1:size(walls.segments, 1)
        wall = walls.segments(i, :);
        plot([wall(1), wall(3)], [wall(2), wall(4)], 'k-', 'LineWidth', 6);
        % Add circular markers at wall endpoints for clarity
        plot(wall(1), wall(2), 'ko', 'MarkerSize', 6, 'MarkerFaceColor', 'k');
        plot(wall(3), wall(4), 'ko', 'MarkerSize', 6, 'MarkerFaceColor', 'k');
    end
    
    % Draw goal with flag
    theta = 0:0.1:2*pi;
    goal_circle_x = goal_pos(1) + goal_radius * cos(theta);
    goal_circle_y = goal_pos(2) + goal_radius * sin(theta);
    fill(goal_circle_x, goal_circle_y, 'k');
    plot(goal_pos(1), goal_pos(2), 'wo', 'MarkerSize', 8, 'MarkerFaceColor', 'w');
    % Add flag
    plot([goal_pos(1), goal_pos(1)], [goal_pos(2), goal_pos(2)+0.5], 'k-', 'LineWidth', 2);
    patch([goal_pos(1), goal_pos(1)+0.3, goal_pos(1)], ...
          [goal_pos(2)+0.5, goal_pos(2)+0.4, goal_pos(2)+0.3], 'red');
    text(goal_pos(1), goal_pos(2)-0.7, 'GOAL', 'HorizontalAlignment', 'center', 'FontWeight', 'bold');
    
    % Draw start (make it bigger and more visible)
    plot(start_pos(1), start_pos(2), 'ro', 'MarkerSize', 20, 'MarkerFaceColor', 'r', 'LineWidth', 2);
    text(start_pos(1), start_pos(2)-0.5, 'START', 'HorizontalAlignment', 'center', 'FontWeight', 'bold');
    
    % Draw path
    if size(path, 1) > 1
        plot(path(:, 1), path(:, 2), 'y-', 'LineWidth', 3);
        plot(path(:, 1), path(:, 2), 'yo', 'MarkerSize', 5);
        
        % Add arrows
        for i = 1:min(10, size(path, 1)-1):size(path, 1)-1
            dx = path(i+1, 1) - path(i, 1);
            dy = path(i+1, 2) - path(i, 2);
            if abs(dx) > 1e-6 || abs(dy) > 1e-6
                quiver(path(i, 1), path(i, 2), dx*0.5, dy*0.5, 0, ...
                       'Color', 'y', 'LineWidth', 2, 'MaxHeadSize', 0.5);
            end
        end
    end
    
    % Draw tile grid overlay (for one tiling)
    for i = 0:tiles_per_dim
        x_line = i * tile_width;
        plot([x_line, x_line], [0, field_height], 'k:', 'LineWidth', 0.5);
    end
    for j = 0:tiles_per_dim
        y_line = j * tile_height;
        plot([0, field_width], [y_line, y_line], 'k:', 'LineWidth', 0.5);
    end
    
    xlabel('X Position', 'FontSize', 12);
    ylabel('Y Position', 'FontSize', 12);
    title(sprintf('Continuous Minigolf with Tile Coding - Path Length: %d steps', size(path, 1) - 1), ...
          'FontSize', 14);
    axis equal;
    axis([0 field_width 0 field_height]);
    grid on;
    
    % Add legend
    h1 = plot(NaN, NaN, 'ko', 'MarkerSize', 10, 'MarkerFaceColor', 'k');
    h2 = plot(NaN, NaN, 'ro', 'MarkerSize', 10, 'MarkerFaceColor', 'r');
    h3 = patch(NaN, NaN, 'blue');
    h4 = plot(NaN, NaN, 'y-', 'LineWidth', 3);
    h5 = plot(NaN, NaN, 'k-', 'LineWidth', 4);
    legend([h1, h2, h3, h4, h5], {'Goal', 'Start', 'Water', 'Path', 'Walls'}, ...
           'Location', 'best');
    
    % Plot 2: Learning curve
    subplot(2, 2, 2);
    window_size = 100;
    % Use filter for moving average
    smoothed_steps = filter(ones(1,window_size)/window_size, 1, episode_steps);
    plot(smoothed_steps, 'b-', 'LineWidth', 2);
    xlabel('Episode', 'FontSize', 12);
    ylabel('Steps to Goal', 'FontSize', 12);
    title('Learning Progress (Steps)', 'FontSize', 12);
    grid on;
    ylim([0 max(smoothed_steps(1000:end))*1.1]);
    
    % Plot 3: Reward curve
    subplot(2, 2, 4);
    smoothed_rewards = filter(ones(1,window_size)/window_size, 1, episode_rewards);
    plot(smoothed_rewards, 'r-', 'LineWidth', 2);
    xlabel('Episode', 'FontSize', 12);
    ylabel('Total Reward', 'FontSize', 12);
    title('Learning Progress (Rewards)', 'FontSize', 12);
    grid on;
    
    sgtitle(sprintf('SARSA(λ) Results - λ=%.1f, %s traces', lambda, trace_type));
end