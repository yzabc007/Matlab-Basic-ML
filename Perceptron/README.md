# Perceptron

The idea of Perceptron can be applied into Linear Regression and Logistic Regression and the differences between typical Perceptron, Linear regression and logistic regression are the activation function and cost function.

For perceptron, the activation function is binary threshold function and the cost function is sum-of-squares function. Since the binary threshold function is non-differentiable, there is no derivation for it. But we can define a special update rule for typical perceptron to optimalize it to simulate the process of gradient descent. Of course, we can use Normal Equation directly.

For linear regression, the activation function is linear function and the cost function is also sum-of-sqaures function. Linear function is differentiable and we can calculate the derivation of the sum-of-squares function (which is the same as the update rule of typical perceptron). Then we can use gradient descent to update cost function. We can also use Normal Equation here because of the sum-of-squares function.

For logistic regression, the activation function is sigmoid function the the cost function is a special designed function: 
![equation](http://www.sciweavers.org/tex2img.php?eq=J%28%5Ctheta%29%20%3D%20%5Cfrac%7B1%7D%7Bm%7D%20%5Csum_%7Bi%3D1%7D%5Em%20%5B-y%5E%7B%28i%29%7Dlog%28h_%7B%5Ctheta%7D%28x%5E%7B%28i%29%7D%29%29-%281-y%5E%7B%28i%29%7D%29log%281-h_%7B%5Ctheta%7D%28x%5E%7B%28i%29%7D%29%29%5D&bc=White&fc=Black&im=jpg&fs=12&ff=arev&edit=0)

The derivation for this cost function is exactly the same as the derivation of sum-of-squares function. 
![equation](http://www.sciweavers.org/tex2img.php?eq=%5Cfrac%7B%5Cpartial%20J%28%5Ctheta%29%7D%7B%5Cpartial%20%5Ctheta_j%7D%3D%5Cfrac%7B1%7D%7Bm%7D%5Csum_%7Bi%3D1%7D%5E%7Bm%7D%20%28h_%7B%5Ctheta%7D%28x%5E%7B%28i%29%7D%29-y%5E%7B%28i%29%7D%29x_j%5E%7B%28i%29%7D&bc=White&fc=Black&im=jpg&fs=12&ff=arev&edit=0)

Notice, this cost function is designed for binary classfication with lable of 1 and 0.  
