1. What tasks has each group member been assigned to for Milestone 2?

For Milestone 2, our group has decided to work on software implementation together, and documentation and 
testing separately. More specifically, we anticipate our current workflow to evolve as follows:
- Blake, Carlos and Aditya will work together in a pair-programming style session to implement the `AutoDiffPy` 
class, which includes writing the `parse_input`, `get_Jacobian` and `get_Derivative` functions. In addition, 
our class will overload basic operations and include important elemental functions (exponential, logarithmic,
trigonometric, etc.) This will enable a working forward mode of automatic differentiation for scalar functions 
of a single input.
- Aditya will work primarily on the testing suite, which will run automatically on Travis CI and Codecov.
- Blake and Carlos will both work on updating our documentation, which is currently written in a Jupyter
notebook to accomodate code blocks for demos.
- After completing their respective sections, all three will reconvene to discuss future features, which will
include parsing vector-valued functions with multiple inputs, building out the gradient descent functionality, 
among other ideas that we may come up with over the course of the development process.

2. What has each group member done since the submission of Milestone 1?

Since Milestone 1, the group received feedback on our original documentation. Our feedback mainly concerned
the handling of vector-valued functions in our two classes and overloading the basic mathematical operations
(addition, subtraction, multiplication, division, and power). Aditya and Carlos worked primarily on responding 
to that feedback in an updated milestone1.ipynb document, while Blake merged the changes and communicated with
the project TF on our updates.

In addition, all three team members coordianted on task allocation for milestone 2 (as outlined in this 
document), and Blake summarized those decisions here. All three members are currently planning out their code
implementation (specifically, our approach for the `parse_input`, `get_Jacobian` and `get_Derivative` methods
of the `AutoDiffPy` class), and we are scheduled to meet for a pair programming session to complete this
implementation together on Saturday, 11/14.
