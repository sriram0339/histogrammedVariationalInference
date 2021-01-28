function plotStem(stem, numTrajs, col, alpha) 
%%figure
hold on
colSpec=['-',col];
for j = 1:numTrajs
    name = [stem,num2str(j)];
    traj = load(name);
  
   % plt1 = plot(traj(:,1), traj(:,2),'.r', 'MarkerSize', 14);
   % plt1.Color(4) = 0.7;
   % hold on;
   
    plt1=plot(traj(:,1), traj(:,2),colSpec, 'LineWidth', 1); 
    plt1.Color(4) = alpha;
    hold on
    %plot(traj(:,1), traj(:,2),'-b', 'LineWidth', 0.05); 
end

end