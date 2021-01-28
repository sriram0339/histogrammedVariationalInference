countPrior = 0;
for j = 1:499
    traj = load(['honeybee-prior--',num2str(j)]);
  
   
    plt1=plot(traj(:,5), traj(:,6),'-b', 'LineWidth', 3); 
    plt1.Color(4) = 0.1;
    hold on
    %plot(traj(:,1), traj(:,2),'-b', 'LineWidth', 0.05); 
    if (traj(end,5) >= traj(end,6))
       countPrior = countPrior + 1 ;
    end
end

fprintf(1, 'Prior Estimate z1 >= z2 : %f\n', countPrior/500);
countPost = 0;
for j = 1:999
    traj = load(['honeybee--',num2str(j)]);
  
   
    plt1=plot(traj(:,5), traj(:,6),'-k', 'LineWidth', 3); 
    plt1.Color(4) = 0.2;
    hold on
    %plot(traj(:,1), traj(:,2),'-b', 'LineWidth', 0.05); 
    if (traj(end,5) >= traj(end,6))
       countPost = countPost + 1 ;
    end
end
fprintf(1, 'Post. Estimate z1 >= z2 : %f\n', countPost/1000);
data = [ 203.7662071288052, 78.76844337739114;
306.6796337583012, 101.81895729607557;
359.22904436109155, 115.22629969404319;
385.3582386922393, 124.35749230837888;
397.5407383580274, 131.66175045217554];     
     hold on
plot(data(:,1), data(:,2), 'ok',  'MarkerSize', 10, 'MarkerFaceColor','r')
