function PlotData(OutType, weights, inputs, targets, predicts)

	if strcmp(OutType, 'linear')
		if size(inputs, 2) == 2
			fprintf('Plotting ... \n');
			figure;
			plot(inputs(:,2), targets, 'rx');
			xlabel('Inputs');
			ylabel('Targets');
			title('Visualization of Linear Regression');
			hold on;
			plot(inputs(:,2), predicts, '-');
			legend('Data', 'Linear regression');
			%legend('Train data fit', 'Linear Regression');
			hold off
		end
	elseif strcmp(OutType, 'logistic')
		if size(inputs, 2) <= 3
			figure;
			pos = find(targets == 1);
			neg = find(targets == 0);
			plot(inputs(pos,2), inputs(pos,3), 'k+', 'LineWidth', 2, 'MarkerSize', 7);
			hold on;
			plot(inputs(neg,2), inputs(neg,3), 'ko', 'MarkerFaceColor', 'y', 'MarkerSize', 7);
			xlabel('X1');
			ylabel('X2');
			legend('Posotive', 'Negative');
			title('Visualization of Logistic Regression');
			hold on;
			plot_x = [min(inputs(:,2)) - 2, max(inputs(:,2)) + 2];
			plot_y = (-weights(2) .* plot_x - weights(1)) ./ weights(3);
			plot(plot_x, plot_y);
			hold off
		end
	end

end