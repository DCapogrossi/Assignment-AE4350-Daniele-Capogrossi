%% Golf Game Reinforcement Learning in MATLAB - Version 4 Modified
% Q-learning with tile coding for continuous state representation
% Course: AE4350 Bio-inspired Intelligence
% Continuous position minigolf with walls and 24-directional movement (every 15 degrees)

clear; clc; close all;

%% Game Environment Setup
% Define continuous field dimensions
field_width = 10;            % Width 
field_height = 8;            % Height 
 
% Start and goal positions (continuous)
goal_pos = [1.0, 7.0];       % Upper-left area
start_pos = [9.0, 1.0];      % Lower-right area
goal_radius = 0.2;           % Radius for reaching the goal

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

%% Tile Coding Setup
num_tilings = 4;                                 % Number of offset tilings
tiles_per_dim = 20;                              % Number of tiles for each dimension
tile_width = field_width / tiles_per_dim;
tile_height = field_height / tiles_per_dim;
num_directions = 36;                             % Directions in which the ball can move (Every 360/n.directions degree)
num_force_levels = 14;                           % Distances that the single shot can cover
num_actions = num_directions * num_force_levels;

% Total number of features (tiles)
total_tiles = num_tilings * tiles_per_dim * tiles_per_dim;
tile_offsets1 = rand(num_tilings, 2) .* [tile_width, tile_height];


% Initialize weight vectors for each action
% Using linear function approximation: Q(s,a) = w(a)' * features(s)
weights1 = zeros(total_tiles, num_actions);

%% Q-Learning Parameters
num_episodes = 20000;    
alpha = 0.1 / num_tilings;     % Step size parameter (accounts for tile coding)
gamma = 0.95;                  % Discount factor
epsilon = 0.6;                 % Start with full exploration
epsilon_decay = 0.9995;        % 0.9997 
epsilon_min = 0.01;            % Lower minimum
Q_min = -1e3;                  % Q-value clipping
Q_max = 1e3; 

% Movement parameters 
force_values =[0.1, 0.2, 0.3, 0.4, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0, 7.0, 9.0];  

%% Training Loop
fprintf('Training Q-learning agent with tile coding...\n');
fprintf('Field dimensions: %.1f x %.1f\n', field_width, field_height);
fprintf('Start position: (%.1f, %.1f)\n', start_pos(1), start_pos(2));
fprintf('Goal position: (%.1f, %.1f) with radius %.1f\n', goal_pos(1), goal_pos(2), goal_radius);
fprintf('Number of tilings: %d\n', num_tilings);
fprintf('Tiles per dimension: %d\n', tiles_per_dim);
fprintf('Total features: %d\n', total_tiles);
fprintf('Number of directions: %d\n', num_directions);
fprintf('Number of force levels: %d\n', num_force_levels);
fprintf('Total actions: %d\n', num_actions);

tic;

% Track performance
episode_steps = zeros(num_episodes, 1);
episode_rewards = zeros(num_episodes, 1);
success_count = 0;

for episode = 1:num_episodes
    % Reset to start position
    ball_pos = start_pos;
    
    % Get initial state features
    features = get_tile_features(ball_pos(1), ball_pos(2), num_tilings, tiles_per_dim, ...
                                tile_width, tile_height,tile_offsets1, field_width, field_height);
    
    % Choose initial action (epsilon-greedy)
    if rand < epsilon
        action = randi(num_actions);
    else
        Q_values = weights1' * features;              % Compute Q-values for all actions
        Q_values = min(max(Q_values, Q_min), Q_max);  % clip Q values
        [~, action] = max(Q_values);                  % Takes the action with highest Q value
    end
    
    done = false;
    step_count = 0;
    max_steps = 1000;    
    total_reward = 0;
    
    while ~done && step_count < max_steps
        step_count = step_count + 1;
        
        % Execute action and get new position
        new_pos = execute_continuous_action(ball_pos, action, ...
                 field_width, field_height, walls, num_directions, num_force_levels, force_values,obstacles);
        
        % Get reward
        reward = get_continuous_reward(ball_pos, new_pos, obstacles, goal_pos, goal_radius);
        total_reward = total_reward + reward;
        
        % Check if episode is done (reached the goal or fell in the water)
        done = check_continuous_done(new_pos, obstacles, goal_pos, goal_radius);
        
        % Get new state features
        new_features = get_tile_features(new_pos(1), new_pos(2), num_tilings, tiles_per_dim, ...
                                        tile_width, tile_height,tile_offsets1, field_width, field_height);
        
        % Choose next action (epsilon-greedy)
        if rand < epsilon
            next_action = randi(num_actions);
        else
            Q_values_next = weights1' * new_features;
            Q_values_next = min(max(Q_values_next, Q_min), Q_max); 
            [~, next_action] = max(Q_values_next);
        end
        
        % Q-learning update with linear function approximation
        Q_current = weights1(:, action)' * features;
        
        if done
            target = reward;  
           if norm(new_pos - goal_pos) <= goal_radius
              success_count = success_count + 1;
           end

        else
            Q_next_max = max(weights1' * new_features);
            target = reward + gamma * Q_next_max;
        end
        
        % Update weights
        delta = target - Q_current;
        weights1(:, action) = weights1(:, action) + alpha * delta * features;
        
        % Update for next iteration
        ball_pos = new_pos;
        features = new_features;
        action = next_action;
    end
    
    % Store episode statistics
    episode_steps(episode) = step_count;
    episode_rewards(episode) = total_reward;
    
    % Decay epsilon
    % epsilon = max(epsilon_min, epsilon - 0.0001);       % linear decay
    epsilon = max(epsilon_min, epsilon * epsilon_decay);  % exponential decay
    
    % Progress indicator
    if mod(episode, 1000) == 0
        avg_steps = mean(episode_steps(max(1,episode-999):episode));
        recent_success = success_count / episode;
        fprintf('Episode %d/%d | Epsilon: %.3f | Avg steps: %.1f | Success rate: %.1f%%\n', ...
                episode, num_episodes, epsilon, avg_steps, recent_success*100);
    end
end

training_time = toc;
fprintf('\nTraining completed in %.2f seconds.\n', training_time);
fprintf('Total successful episodes: %d / %d (%.1f%%)\n', ...
        success_count, num_episodes, success_count/num_episodes*100);
save('minigolf_weights1.mat', 'weights1');
save('tile_offsets1.mat', 'tile_offsets1');
fprintf('Weights saved to minigolf_weights.mat\n');

%% Test the Learned Policy
fprintf('\nTesting learned policy:\n');

if exist('minigolf_weights1.mat', 'file')
    load('minigolf_weights1.mat', 'weights1');
    fprintf('Loaded trained weights from minigolf_weights.mat\n');
else
    warning('Trained weights not found. Running with untrained policy.');
end

% Generate optimal path
ball_pos = start_pos;
path = ball_pos;
step = 0;
max_test_steps = 500;

fprintf('Starting at position (%.1f, %.1f)\n', ball_pos(1), ball_pos(2));

while norm(ball_pos - goal_pos) > goal_radius && step < max_test_steps
    % Get state features
    features = get_tile_features(ball_pos(1), ball_pos(2), num_tilings, tiles_per_dim, ...
                                tile_width, tile_height,tile_offsets1, field_width, field_height);
    
    % Choose best action
    Q_values = weights1' * features;
    [max_Q, best_action] = max(Q_values);
    
 
    
    % Execute action
    new_pos = execute_continuous_action(ball_pos, best_action, ...
              field_width, field_height, walls, num_directions, num_force_levels, force_values,obstacles);
    
    % Check if stuck
    if norm(new_pos - ball_pos) < 1e-6
        fprintf('Warning: Agent stuck at position (%.2f, %.2f)\n', ball_pos(1), ball_pos(2));
        break;
    end
    
    ball_pos = new_pos;
    path = [path; ball_pos];
    step = step + 1;
end

if norm(ball_pos - goal_pos) <= goal_radius
    fprintf('Goal reached in %d steps!\n', step);
else
    fprintf('Failed to reach goal in %d steps.\n', step);
end
fprintf('Final position: (%.2f, %.2f)\n', ball_pos(1), ball_pos(2));
fprintf('Distance to goal: %.2f\n', norm(ball_pos - goal_pos));

%% Visualize Results
visualize_continuous_results(field_width, field_height, path, obstacles, goal_pos, goal_radius, ...
                           start_pos, walls, tiles_per_dim, ...
                           tile_width, tile_height, episode_steps, episode_rewards);



%% Helper Functions

function features = get_tile_features(x, y, num_tilings, tiles_per_dim, tile_width, tile_height,tile_offsets, field_width, field_height)
    % Get binary feature vector for tile coding
    % Returns sparse binary vector with num_tilings active features
    
    features = zeros(num_tilings * tiles_per_dim * tiles_per_dim, 1);
    
     for tiling = 1:num_tilings
        % Apply offset for this tiling
        offset_pos = [x, y] - tile_offsets(tiling, :);

        % Wrap around for offset positions
        offset_pos(1) = mod(offset_pos(1), field_width);
        offset_pos(2) = mod(offset_pos(2), field_height);

        % Convert to tile indices
        tile_x = floor(offset_pos(1) / tile_width);
        tile_y = floor(offset_pos(2) / tile_height);

        % Ensure indices are within bounds
        tile_x = max(0, min(tiles_per_dim - 1, tile_x));
        tile_y = max(0, min(tiles_per_dim - 1, tile_y));

        % Convert to linear index
        tile_index = tile_y * tiles_per_dim + tile_x + 1;

        % Set feature for this tiling
        feature_index = (tiling - 1) * tiles_per_dim * tiles_per_dim + tile_index;
        features(feature_index) = 1;
     end
      
end

function new_pos = execute_continuous_action(pos, action_index, ...
    field_width, field_height, walls, num_directions, num_force_levels, force_values,obstacles)

    [dir_idx, force] = decode_action(action_index, num_directions, num_force_levels, force_values);
    angle = (dir_idx - 1) * (2 * pi / num_directions);
    
    % Calculate movement components
    dx = force * cos(angle);
    dy = force * sin(angle);
    
    intended_pos = pos + [dx, dy];

    % Clamp to field boundaries
    intended_pos(1) = max(0.2, min(intended_pos(1), field_width - 0.2));
    intended_pos(2) = max(0.2, min(intended_pos(2), field_height - 0.2));

   if check_wall_collision(pos, intended_pos, walls) || crosses_water_hazard(pos, intended_pos, obstacles)
    new_pos = pos;
   else
    new_pos = intended_pos;
   end

end

function [dir_idx, force] = decode_action(action_index, num_directions, num_force_levels, force_values)
    % Decode action index into direction and force components
    % Actions are numbered from 1 to num_directions * num_force_levels
    
    % Convert to 0-based indexing
    action_idx_0 = action_index - 1;
    
    % Decode direction and force level
    dir_idx = mod(action_idx_0, num_directions) + 1;
    force_level = floor(action_idx_0 / num_directions) + 1;
    
    % Get actual force value
    force = force_values(force_level);
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


     % Distance-based shaping (closer = better)
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
    
    % Agent in water hazard
    for i = 1:size(obstacles, 1)
        if pos(1) >= obstacles(i, 1) && pos(1) <= obstacles(i, 3) && ...
           pos(2) >= obstacles(i, 2) && pos(2) <= obstacles(i, 4)
            done = true;
            return;
        end
    end
    
    done = false;
end

function crossed = crosses_water_hazard(pos1, pos2, obstacles)
    crossed = false;
    
     for i = 1:size(obstacles, 1)
        % Get water bounds
        x_min = obstacles(i, 1);
        y_min = obstacles(i, 2);
        x_max = obstacles(i, 3);
        y_max = obstacles(i, 4);
        
        % Define corners of the rectangle
        corners = [
            x_min, y_min;
            x_max, y_min;
            x_max, y_max;
            x_min, y_max
        ];
        
        % Define edges of rectangle
        edges = [
            corners(1, :), corners(2, :);
            corners(2, :), corners(3, :);
            corners(3, :), corners(4, :);
            corners(4, :), corners(1, :)
        ];
        
        % Check intersection with each edge
        for j = 1:4
            if line_segments_intersect( ...
                    pos1(1), pos1(2), pos2(1), pos2(2), ...
                    edges(j,1), edges(j,2), edges(j,3), edges(j,4))
                crossed = true;
                return;
            end
        end
        
        % Optional: Also check if mid-point is inside the obstacle (overlapping edge cases)
        midpoint = (pos1 + pos2) / 2;
        if midpoint(1) >= x_min && midpoint(1) <= x_max && ...
           midpoint(2) >= y_min && midpoint(2) <= y_max
            crossed = true;
            return;
        end
    end
end

function intersect = line_segments_intersect(x1, y1, x2, y2, x3, y3, x4, y4)
    % Robust line segment intersection that handles edge cases
    
    % Calculate direction vectors
    dx1 = x2 - x1; dy1 = y2 - y1;
    dx2 = x4 - x3; dy2 = y4 - y3;
    
    % Calculate determinant
    det = dx1 * dy2 - dy1 * dx2;
    
    % Check if lines are parallel
    if abs(det) < 1e-10
        % Lines are parallel, check if they are collinear and overlapping
        % Calculate cross product to check collinearity
        cross1 = (x3 - x1) * dy1 - (y3 - y1) * dx1;
        if abs(cross1) < 1e-10
            % Lines are collinear, check if segments overlap
            % Project points onto line direction
            if abs(dx1) > abs(dy1)
                % Project onto x-axis
                t1 = 0;
                t2 = 1;
                t3 = (x3 - x1) / dx1;
                t4 = (x4 - x1) / dx1;
            else
                % Project onto y-axis
                t1 = 0;
                t2 = 1;
                t3 = (y3 - y1) / dy1;
                t4 = (y4 - y1) / dy1;
            end
            
            % Sort t3 and t4
            if t3 > t4
                temp = t3;
                t3 = t4;
                t4 = temp;
            end
            
            % Check overlap
            intersect = t3 <= 1 && t4 >= 0;
        else
            intersect = false;
        end
        return;
    end
    
    % Calculate parameters
    t1 = ((x3 - x1) * dy2 - (y3 - y1) * dx2) / det;
    t2 = ((x3 - x1) * dy1 - (y3 - y1) * dx1) / det;
    
    % Check if intersection occurs within both segments
    % Use small epsilon for numerical stability
    eps = 1e-10;
    intersect = t1 >= -eps && t1 <= 1 + eps && t2 >= -eps && t2 <= 1 + eps;
end

function collision = check_wall_collision(pos1, pos2, walls)
    % Check if movement from pos1 to pos2 collides with any wall
    collision = false;
    
    if ~isfield(walls, 'segments') || isempty(walls.segments)
        return;
    end
    
    for i = 1:size(walls.segments, 1)
        wall_start = walls.segments(i, 1:2);
        wall_end = walls.segments(i, 3:4);
        
        if line_segments_intersect(pos1(1), pos1(2), pos2(1), pos2(2), wall_start(1),wall_start(2), wall_end(1),wall_end(2))
            collision = true;
            return;
        end
    end
end


function visualize_continuous_results(field_width, field_height, path, obstacles, goal_pos, goal_radius, ...
                                    start_pos, walls, tiles_per_dim, ...
                                    tile_width, tile_height, episode_steps, episode_rewards)
    
    figure('Position', [50, 50, 1400, 800]);
    
    % Plot 1: Golf course with path
    subplot(2, 2, [1, 3]);
    
    % Draw field background
    rectangle('Position', [0, 0, field_width, field_height], ...
              'FaceColor', [0.2 0.8 0.2], 'EdgeColor', 'k', 'LineWidth', 2);
    hold on;
    
    % Draw water hazards
    for i = 1:size(obstacles, 1)
        rectangle('Position', [obstacles(i, 1), obstacles(i, 2), ...
                              obstacles(i, 3) - obstacles(i, 1), ...
                              obstacles(i, 4) - obstacles(i, 2)], ...
                  'FaceColor', [0.1 0.4 0.8], 'EdgeColor', 'k');
    end
    
    % Draw walls with end markers
    for i = 1:size(walls.segments, 1)
        wall = walls.segments(i, :);
        plot([wall(1), wall(3)], [wall(2), wall(4)], 'k-', 'LineWidth', 4);
        % Add circular markers at wall endpoints for clarity
        plot(wall(1), wall(2), 'ko', 'MarkerSize', 4, 'MarkerFaceColor', 'k');
        plot(wall(3), wall(4), 'ko', 'MarkerSize', 4, 'MarkerFaceColor', 'k');
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
    
    % Draw start
    plot(start_pos(1), start_pos(2), 'ro', 'MarkerSize', 10, 'MarkerFaceColor', 'r', 'LineWidth', 2);
    text(start_pos(1), start_pos(2)+0.5, 'START', 'HorizontalAlignment', 'center', 'FontWeight', 'bold');
    
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
    
    sgtitle('Minigolf RL with Tile Coding - Continuous State Space', 'FontSize', 16);
end