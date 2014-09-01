function [J grad] = nnCostFunction(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X, y, lambda)
%NNCOSTFUNCTION Implements the neural network cost function for a two layer
%neural network which performs classification
%   [J grad] = NNCOSTFUNCTON(nn_params, hidden_layer_size, num_labels, ...
%   X, y, lambda) computes the cost and gradient of the neural network. The
%   parameters for the neural network are "unrolled" into the vector
%   nn_params and need to be converted back into the weight matrices. 
% 
%   The returned parameter grad should be a "unrolled" vector of the
%   partial derivatives of the neural network.
%

% Reshape nn_params back into the parameters Theta1 and Theta2, the weight matrices
% for our 2 layer neural network
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

% Setup some useful variables
m = size(X, 1);
X = [ones(m,1) X]; 
temp3 = sigmoid(X*Theta1');
n=size(temp3,1);
temp3=[ones(n,1) temp3];
   
% You need to return the following variables correctly 
J = 0;
temp =eye(num_labels);
temp2 =((-temp(y,:).*log(sigmoid((temp3)*Theta2'))-(-temp(y,:).+1).*log(-sigmoid(temp3*Theta2').+1))./m);
J=sum(sum(temp2))+(sum(sum((Theta1(:,2:(input_layer_size+1)).^2)))+sum(sum((Theta2(:,2:(hidden_layer_size+1)).^2))))*(lambda/(2*m));
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));
%y(4000);

%finald2 =zeros((size(Theta2,1)),(size(Theta2,2)-1));
%finald1 =zeros((size(Theta1,1)),(size(Theta1,2)-1));
%size(Theta1);
%for i = 1:m
%	a0i=X(i,:);
%	z2=a0i*Theta1';
%	a1i=sigmoid(a0i*Theta1');
%	n=size(a1i,1);
%	a1i=[1 a1i];
%	outputi  = a1i*Theta2';
%	size(outputi);
%	qwer = 1:num_labels;
%	qwer==y(i);
%	delta3=outputi.-qwer;
%	sani = Theta2'*delta3';
%	z2=[1 z2];
%	delta2 =  (sani)'.*sigmoidGradient(z2);
%	delta2 = delta2(2:end);
%	finald2 = finald2 + delta3'*(sigmoid(a0i*Theta1'));
%	shiv =a0i(2:end);
%	finald1 = finald1 + delta2'*(shiv);
%	%size(finald);
	


%endfor
%Theta2_grad = finald2./m;
%Theta1_grad = finald1./m;
% ====================== YOUR CODE HERE ======================
% Instructions: You should complete the code by working through the
%               following parts.
%
% Part 1: Feedforward the neural network and return the cost in the
%         variable J. After implementing Part 1, you can verify that your
%         cost function computation is correct by verifying the cost
%         computed in ex4.m
%
% Part 2: Implement the backpropagation algorithm to compute the gradients
%         Theta1_grad and Theta2_grad. You should return the partial derivatives of
%         the cost function with respect to Theta1 and Theta2 in Theta1_grad and
%         Theta2_grad, respectively. After implementing Part 2, you can check
%         that your implementation is correct by running checkNNGradients
%
%         Note: The vector y passed into the function is a vector of labels
%               containing values from 1..K. You need to map this vector into a 
%               binary vector of 1's and 0's to be used with the neural network
%               cost function.
%
%         Hint: We recommend implementing backpropagation using a for-loop
%               over the training examples if you are implementing it for the 
%               first time.
%
% Part 3: Implement regularization with the cost function and gradients.
%
%         Hint: You can implement this around the code for
%               backpropagation. That is, you can compute the gradients for
%               the regularization separately and then add them to Theta1_grad
%               and Theta2_grad from Part 2.
%

% ===========================		Part 2  	===========================
DELTA1 = zeros(hidden_layer_size, input_layer_size+1);
DELTA2 = zeros(num_labels, hidden_layer_size+1);
for i=1:m
	%	Compute activations
	a1 = X(i,:)';
	z2 = Theta1 * a1;
	a2 = sigmoid(z2);
	a2 = [1;a2];
	z3 = Theta2 * a2;
	a3 = sigmoid(z3);

	% Compute delta (output layer)
	yk = zeros(num_labels,1);
	yk( y(i) ) = 1;
	delta3 = a3 - yk;

	% Compute delta (hidden layer) 
	z2 = [ 1;z2 ];
	delta2 = (Theta2'*delta3) .* ( sigmoidGradient(z2) );

	% Accumulate the gradient
	delta2 = delta2(2:end);
	DELTA1 = DELTA1 + delta2 * a1';
	DELTA2 = DELTA2 + delta3 * a2';
end;

Theta1_grad = ( 1/m ) * ( DELTA1 ) + (lambda/m)*Theta1;
Theta2_grad = ( 1/m ) * ( DELTA2 ) + (lambda/m)*Theta2;
Theta1_grad(:,1)-=(lambda/m)*Theta1(:,1);
Theta2_grad(:,1)-=(lambda/m)*Theta2(:,1);


















% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
