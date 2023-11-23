#align(center, text(17pt)[
  The search of the Third Integral of Motion: \
    Some Numerical Investigations
])

#align(center)[
  #set par(justify: false)
  Victor Velev \
  velev\@wisc.edu
]

#show: rest => columns(2, rest)


= Introduction

In the early 60's astronomers were up against a wall. Careful measurements of
the motion of nearby starts in the galaxy allowed for computing averages of the
observed motions. Unfortunately, the averages where way of from the theoretical
predictions. Several assumptions where made such as the axisymmetricity of the galaxy,
which implied an axisymmetric central potential. Perhaps the galaxy has non-axissymmetric
components of the potential which were unfairly neglected? It turned out that the problem
was much deeper. The understanding of motion was wrong.

Consider the phase-space distribution function $f(arrow(x), arrow(p))$, which gives the
probability density of finding a star at a position $arrow(x)$ with momentum $arrow(p)$.
In terms of $f$, the statistical average of any dynamical quantity $w$ over some volume
of space $V$ is just

#align(center)[$angle.l w angle.r#sub[V] = integral#sub[V] f w$]

Theoretical considerations concluded that the velocity dispersion in the radial direction
should be about the same as the velocity dispersion in the direction orthogonal to the galactic plane.
In cylindrical polar coordinates $(r, theta, z)$:

$
sigma#sub[r] = angle.l (dot(r) - angle.l dot(r) angle.r)^2 angle.r ^(1/2) \
sigma#sub[z] = angle.l (dot(z) - angle.l dot(z) angle.r)^2 angle.r ^(1/2)
$

And the conclusion being $sigma#sub[r] approx sigma#sub[z]$. The measurements
showed $sigma_r approx 2 sigma_z$.

In physics, questions about motions are solved with the help of _integrals of motion_
\- constants which are conserved throughout the trajectory of the moving body. Because of the
vastness of the galaxy, it is not unreasonable to assume that the distribution of stars in the galaxy
does not change much with time, or changes only very slowly. That is to say, close encounters
with other stars are very rare. Therefore, we can consider each individual star as moving
in a time-independent, axisymmetric central potential. In a system of cylindrical polar coordinates
$r, theta, z$ the Hamiltonian is

$a#sub[theta]$

$T + V = 1/(2m) [p#sub[r]^2 + (p#sub[theta])^2/r^2 + p#sub[z]] + V(r, z)$

// this gives us $5$ \textit{integrals of motion}; that is, five independent functions:
//
// $$I_j(r, theta, z, p_r, p_theta, p_z)$$
//
// such that
//
// $$I_j = C_j$$
//
// for some five constants $C_1, dots.h, C_5$. Each equation represents a hypersurface in the phase-space,
// and the trajectory is the intersection of the five hypersurfaces.

// However, some integrals of motion are better then others. Some integrals of motion essentially cover the
// whole phase-space, therefore, they do not restrict the motion of the body in any way (i.e. they do no bring
// new information about the motion of the body). Since the Hamiltonian has no explicit dependence either
// on time, nor on $\theta$, theory tells us that there are $2$ significant integrals of motion, namely the
// total energy $E$ and the conjugate angular momemntum $p_{\theta}$. Jean's theorem asserts that the distribution
// function $f$ can be rewritten as a function of the significant integrals of motion
// $$f'(E, p_{\theta}) = f(\vec{x}, \vec{p})$$
// Therefore, one can conclude that the phase-space to be explored
// by the stars is restricted only by $E$ and $p_{\theta}$. Observe further that in the equation for $E$ (the
// Hamiltonian) $p_{z}$ and $p_{r}$ appear in the same way. Thus, it makes sense that any statistical
// average for $p_{z}$ and $p_{r}$ computed using $f$ would be the same:
// $$\sigma_{z} = \sigma_{r}$$
// This is not what was observed, maybe we've missed a \textit{third integral of motion} that constrains
// the exploration of the phase-space. While one can approach the search for such an integral analytically,
// I am not familiar with success stories embarking on that route. Hénon and Heiles in their influential 1964
// paper approach the problem through numerical experiments. In this paper I would like to build on top of their
// work, verifying the conclusions they obtained, applying the developed methods to the Restricted 3-body Problem,
// carrying out a qualitative comparison between the two settings, while in the meantime observing the performance
// of the various numerical methods covered in class.

= Related Work
#lorem(200)
