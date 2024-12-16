From: https://graphics.stanford.edu/~mdfisher/cloth.html

# Equations of Motion
## Mass-Spring Model
The equations of motion for a system govern the motion of the system. One of the first (and simplest) cloth models is as follows: consider the sheet of cloth. Divide it up into a series of approximately evenly spaced masses M. Connect nearby masses by a spring, and use Hooke's Law and Newton's 2nd Law as the equations of motion. Various additions, such as spring damping or angular springs, can be made. A mesh structure proves invaluable for storing the cloth and performing the simulation directly on it. Each vertex can store all of its own local information (velocity, position, forces, etc.) and when it comes time to render, the face information allows for immediate rendering as a normal mesh.

In the cloth structure seen above, the red nodes are vertices, and the black lines are springs. The diagonal springs are necessary to resist collapse of the face; it ensures that the entire cloth does not decompose into a straight line. The blue represents the mesh faces. Looking at the equations of motion:

$$
\begin{aligned}
& F_{\text {net }}(v)=\mathrm{Mg}+F_{\text {wind }}+F_{\text {air resistance }}-\sum_{\text {Springs } \in v} k\left(x_{\text {current }}-x_{\text {rest }}\right)=\mathrm{Ma} \\
& M=\text { mass of vertex } \\
& g=\text { gravity vector }=(0,-9.8,0) \\
& k=\text { spring constant } \\
& x_{\text {current }}=\text { current length of spring } \\
& x_{\text {rest }}=\text { rest (initial) length of spring } \\
& F_{\text {wind }}=\text { wind vector } \\
& F_{\text {air resistance }}=-a * \text { velocity }(v)^2
\end{aligned}
$$

To determine $M$, a simple constant (say, 1) is fine for all vertices. To be more accurate, you should compute the area of each triangle, and assign $1 / 3$ rd of it towards the mass of each incident vertex; this way the mass of the entire cloth is the total area of all the triangles times the mass density of the cloth. The gravity vector can also be an arbitrary vector; if all distance units were meters, time was measured in seconds, and we were on the surface of the earth and " $y$ " was the "up/down" vector, $(0,-9.8,0$ ) would be the correct " $g$ ". $X$ (current) is just the current length of the spring, and X (rest), the spring's rest length, needs to be stored in each spring structure. $F($ wind $)$ can just be some globally varying constant function, say $\left(\sin \left(x^* y^* t\right), \cos \left(z^* t\right), \sin \left(\cos \left(5^* x^* y^* z\right)\right)\right.$. a is a simple constant determined by the properties of the surrounding fluid (usually air,) but it can also be used to achieve certain cloth effects, and can help with the numeric stability of the cloth. k is a very important constant; if too low, the cloth will sag unrealistically:


On the other hand, if $k$ is chosen too large, the system will be unrealistically tight (retain its original shape.) Worse yet, the system will be more and more "stiff" the larger $k$ is chosen; this is a mathematical term for explosive. Without careful attention to the integration method, the system will gain energy as a function of time, and the system will explode with vertices tending towards infinity.

## Elasticity Model
The mass-spring model above has several shortcommings. Mostly, it is not very physically correct, so attempting to use physically accurate constants is generally unsuccessful. Furthermore it requires guessing arbitrarily which vertices should be connected to which by a spring, and choosing k such that increasing the resolution of the grid leads to a system with similar characteristics can be tricky. A more accurate model, based on integrating an energy over the surface of the cloth, consider energy terms such as:
- Triangles and edges resist changes in their size, and compressing or expanding them requires energy
- Edges resist bending, so bending the two faces adjacent away from the inital bend of this edge ( 0 for a planar initial condition) requires energy
- Triangles resist deformation (in addition to resisting changes in size.) So attempting to shear or otherwise deform a triangle requires energy

We imagine a giant vector, S, representing every important variable in the system (position and velocities of all the vertices, although potentially there could be more degrees of freedom.) Given energy as a function of the current state of the system, $E(S)$, the equation of motion for a single vertex at position ( $x, y$, z) is then rather simple:

$$
\mathbf{F}_{\mathrm{net}}=\left\{\frac{\partial \mathrm{E}(\mathbf{S})}{\partial \mathbf{x}}, \frac{\partial \mathrm{E}(\mathbf{S})}{\partial \mathbf{y}}, \frac{\partial \mathrm{E}(\mathbf{S})}{\partial \mathrm{z}}\right\}
$$


Evaluating this, however, is not so simple. Generally this derivative must be computed analytically. Suppose we attempted to compute the derivative numerically; we consider the state variable constant, reducing our energy $E(s)$ to $E(x)$. We then say:

$$
\frac{\partial \mathrm{E}(\mathrm{x})}{\partial \mathrm{x}}=\frac{\mathrm{E}(\mathrm{x}+\Delta \mathrm{x})-\mathrm{E}(\mathrm{x})}{\Delta \mathrm{x}}
$$


But evaluating the energy $E(S)$ takes a long time; we must iterate over all the vertices, faces, and edges, summing the energy of each one. But we might notice that the effect of $x$ on the energy depends on a very local region (all the incident edges and faces, called the one-ring of the vertex.) So to keep our algorithm $O(n)$ when doing the derivative numerically, we must make sure that we compute $E(x)$ by only considering the energy of the one-ring of the vertex in question.

In general, this method can be very challenging to implement, and although it is physically much more sound, in practice the results are in some ways better and in some ways worse than the mass-spring model. This method of cloth animation, including the derivation of the energy terms, is discussed thoroughly in this paper.

# Integrators
After we decide on the cloth model, we need a method to integrate the equation of motion. Assuming our model is newtonian, we have at every vertex defined a position and velocity at each time step $t$, and our equation of motion tells us dv/dt, or the acceleration of each vertex at time $t$, and we want to know the position and velocity at the next time step.

## Euler's Method (Explicit)
The simplest method for integrating our equations is Euler's method. It goes like this:

$$
\begin{aligned}
& v_{t+\Delta t}=v_t+\Delta t\left(\frac{d v}{d t}\right)_t \\
& x_{t+\Delta t}=x_t+\Delta t v_t
\end{aligned}
$$
The t subscript on dv/dt means "dv/dt evaluated at time t" (as opposed to say, dv/dt at the previous or the next time step.) Delta t refers to the timestep we're taking (smaller time step means more accurate results but slower computation times.) We can derive the above method quite simply:



$$
\begin{aligned}
& \mathrm{v}_{\mathrm{t}}=\frac{\mathrm{dx}}{\mathrm{dt}}=\frac{\Delta \mathrm{x}}{\Delta \mathrm{t}} \\
& \Delta \mathrm{x}=\Delta \mathrm{t} \mathrm{v}_{\mathrm{t}} \\
& \mathrm{x}_{\mathrm{t}+\Delta \mathrm{t}}-\mathrm{x}_{\mathrm{t}}=\Delta \mathrm{t} \mathrm{v}_{\mathrm{t}}
\end{aligned}
$$


And likewise for the velocity term. This method is very simple to implement, but it has the disadvantage that for most systems, it has a large amount of positive feedback, and tends to cause all variables to rapidly increase to ridiculous values, no matter how small the time step. This explosive behavior is characteristic of all explicit integrators; the term explicit just means that the state at $(\mathrm{t}+1)$ is evaluated by just considering the state at time $(\mathrm{t})$. The explosive behavior is arguably the most frustrating aspect of all simulations, and a lot of work goes into dealing with this problem. In cloth, the first way around this is to put enough damping in the system (spring damping, air resistance, velocity damping, etc.) that in a single time step energy decreases the naturally, and the feedback will never occur. Another way is to use one of a vast array of add-ons to the cloth model (discussed later) that prevent the explosive behavior. The only way to solve this problem without altering the model is to use an implicit integrator.

## Runge Kutta (Explicit)
Euler's method is not only explosive, it is very inaccurate. As you decrease the time step, the error decreases proportionally. It is possible to use higher-order terms of the derivative to create a much more accurate integrator. There are many such methods, one of the most widely-used of which is called RungeKutta. The N -th order Runge-kutta algorithm considers derivative terms up to the N -th order. For various reasons, 4th order is considered optimal, since it gurantees the integrator error decreases proportional to the fourth power of the time step (and it is not true that 5 -th order is proportional to the fifth power of the time step.) The algorithm is also excellent because it still only needs the first derivative; higher derivatives are efficently computed numerically just by knowing how to compute the first derivative. While very accurate, it takes slightly longer to compute, and still suffers from all the problems of explicit models, so even if it takes the system longer to explode, in general it still will.

## Verlet Algorithm (Explicit)
The Verlet integration algorithm is such an explicit model with the very interesting propety that it does not need to know anything about the velocity; it computes this internally via looking at the position at both the current and previous time step.

$$
x_{t+\Delta t}=2 x_t-x_{t-\Delta t}+\left(\frac{d v}{d t}\right)_t(\Delta t)^2
$$


Another wonderful aspect of this algorithm is that like 4th order Runge-Kutta, it is 4th order accurate. Because it is quite accurate, easy to implement, and does not need the velocity terms, it is my favorite explicit model and the one I use in all my cloth models.

## Euler's Method (Implicit)
Unlike an explicit integrator, an implicit integrator uses the state variable at the current time step and the derivative at the next time step to compute the state variable at the next time step. There is a perfect implicit analogy to Euler's method; in our model, we have the following equations:

$$
\begin{aligned}
& v_{t+\Delta t}=v_t+\Delta t\left(\frac{d v}{d t}\right)_{t+\Delta t} \\
& x_{t+\Delta t}=x_t+\Delta t v_{t+\Delta t}
\end{aligned}
$$


The only change is the subscript from "t" to " $t+d t$ ". We notice quickly that we only need to deal with the $v(t+1)$ term; the $x(t+1)$ term can then be trivially computed using this new velocity. But computing this new velocity can be extremely tricky. In either the mass-spring or elasticity model, this requires the following: consider the big state vector $S$ (all the velocities and positions in the system) as a $6 \mathrm{n} \times 1$ matrix, where n is the number of vertices. Linearize the equations of motion so we can represent $d v / d t$ at time $t$ as follows:

$$
\left(\frac{\mathrm{dv}}{\mathrm{dt}}\right)_{\mathrm{t}}=\mathrm{QS}_{\mathrm{t}}
$$


Where $Q$ is a giant $3 n$ by $6 n$ matrix representing the linear relationship between the change in velocities and the state of the system. We would then substitute this into the implicit euler equation for dv/dt, and solve for the velocity at the new time step. However, this would involve inverting the massive matrix $Q$ (which is thankfully very sparse, assuming not every vertex is connected to every other vertex.) This is generally accomplished with linear conjugate gradient descent. If you're interested in implementing this, see this paper (the same paper as above.) Overall, this algorithm is very complicated to implement for cloth models, but because of the negative feedback, the energy in the system tends to decrease, rather than increase explosively. This enables you to take much larger time steps and remain stable, and avoids the necessity of excessive damping terms. However, it is very time consuming to take each time step, requires convergence of a matrix inversion algorithm, and because such large time steps are taken and the relationship between error and time step is linear (as with Explicit Euler's,) the algorithm is a very bad approximation of the underlying motion we are integrating.

## Higher order implicit algorithms
One might expect that there is an implicit analogy to Runge-Kutta. While such an analogy exists, and has both the advantage of good errors and a stable system, I have never seen it applied to cloth models; the 1st-order implicit version is sufficently complicated for anyone.

## Symplectic Euler's Method (Semi-Implicit)
Many algorithms exist which are compromises between implicit and explicit models. A simple one is called Symplectic Euler's Method. It's equation of motion is:
$$
\begin{aligned}
& v_{t+\Delta t}=v_t+\Delta t\left(\frac{d v}{d t}\right)_t \\
& x_{t+\Delta t}=x_t+\Delta t v_{t+\Delta t}
\end{aligned}
$$

It is called semi-implicit because it computes the velocity explicitly, but the new position implicitly. This helps reduce the feedback (positive or negative) and can greatly improve stability, at no cost in algorithmic complexity. It also has the powerful advantage that in many systems it conserves energy on average. However, it does not conserve phase or oscillatory motion: in cloth, this results in strange out-of-phase circulations occurring all over the mesh, so in general this algorithm is not a good choice for cloth. Higher order symplectic models exist, but they have similar properties, except for the higher order accuracy.
