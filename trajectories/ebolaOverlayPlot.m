countPrior = 0;
for j = 1:499
    traj = load(['ebola-prior--',num2str(j)]);
  
   
    plt1=plot(traj(:,1), traj(:,4),'-b', 'LineWidth', 5); 
    plt1.Color(4) = 0.01;
    hold on
    %plot(traj(:,1), traj(:,2),'-b', 'LineWidth', 0.05); 
end
countPost = 0;
for j = 1:999
    traj = load(['ebola--',num2str(j)]);
    plt1=plot(traj(:,1), traj(:,4),'-k', 'LineWidth', 3); 
    plt1.Color(4) = 0.05;
    hold on
    %plot(traj(:,1), traj(:,2),'-b', 'LineWidth', 0.05); 
end
data = [ 
    6,0.11;
    11, 0.13;
     16 0.11;
     26  0.07
     ];     
     hold on
plot(data(:,1), data(:,2), 'ok',  'MarkerSize', 10, 'MarkerFaceColor','r')
