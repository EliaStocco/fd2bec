Dependency map
==============

The diagrams are generated from imports between modules in ``fd2bec``. Core
modules are blue and command-line subsystems are yellow. They are split into
three views so that the dependency directions remain readable as the package
grows.

.. graphviz:: generated/core_dependencies.dot
   :align: center
   :caption: Dependencies within the core package.

.. graphviz:: generated/workflow_cli_dependencies.dot
   :align: center
   :caption: Scientific workflow commands and the subsystems they import.

.. graphviz:: generated/utility_cli_dependencies.dot
   :align: center
   :caption: Utility and data commands and the subsystems they import.

.. raw:: latex

   \FloatBarrier

.. include:: generated/dependencies.inc
