function overlayObservations(filename)

traj = load(filename);
hold on
plot(traj(:,1), traj(:,2), 'ok',  'MarkerSize', 10, 'MarkerFaceColor','r')
hold on
%plot(traj(:,1), traj(:,2),'xk', 'MarkerSize', 10)

end