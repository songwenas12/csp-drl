# Copyright 2010-2017 Google
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import print_function

import collections

from ortools.sat.python import cp_model


def MinimalCpSat():
  # Creates the model.
  model = cp_model.CpModel()
# Creates the variables.
  num_vals = 3
  x = model.NewIntVar(0, num_vals - 1, "x")
  y = model.NewIntVar(0, num_vals - 1, "y")
  z = model.NewIntVar(0, num_vals - 1, "z")
  # Create the constraints.
  model.Add(x != y)

  # Create a solver and solve.
  solver = cp_model.CpSolver()
  status = solver.Solve(model)

  if status == cp_model.MODEL_SAT:
    print("x = %i" % solver.Value(x))
    print("y = %i" % solver.Value(y))
    print("z = %i" % solver.Value(z))


class VarArraySolutionPrinter(cp_model.CpSolverSolutionCallback):
  """Print intermediate solutions."""

  def __init__(self, variables):
    self.__variables = variables
    self.__solution_count = 0

  def NewSolution(self):
    self.__solution_count += 1
    for v in self.__variables:
      print('%s=%i' % (v, self.Value(v)), end = ' ')
    print()

  def SolutionCount(self):
    return self.__solution_count


def MinimalCpSatAllSolutions():
  # Creates the model.
  model = cp_model.CpModel()
# Creates the variables.
  num_vals = 3
  x = model.NewIntVar(0, num_vals - 1, "x")
  y = model.NewIntVar(0, num_vals - 1, "y")
  z = model.NewIntVar(0, num_vals - 1, "z")
  # Create the constraints.
  model.Add(x != y)

  # Create a solver and solve.
  solver = cp_model.CpSolver()
  solution_printer = VarArraySolutionPrinter([x, y, z])
  status = solver.SearchForAllSolutions(model, solution_printer)

  print('Number of solutions found: %i' % solution_printer.SolutionCount())


def SolvingLinearProblem():
  # Create a model.
  model = cp_model.CpModel()

  # x and y are integer non-negative variables.
  x = model.NewIntVar(0, 17, 'x')
  y = model.NewIntVar(0, 17, 'y')
  model.Add(2*x + 14*y <= 35)
  model.Add(2*x <= 7)
  obj_var = model.NewIntVar(0, 1000, "obj_var")
  model.Add(obj_var == x + 10*y)
  objective = model.Maximize(obj_var)

  # Create a solver and solve.
  solver = cp_model.CpSolver()
  status = solver.Solve(model)
  if status == cp_model.OPTIMAL:
    print("Objective value: %i" % solver.ObjectiveValue())
    print()
    print('x= %i' %  solver.Value(x))
    print('y= %i' % solver.Value(y))


def MinimalJobShop():
  # Create the model.
  model = cp_model.CpModel()

  machines_count = 3
  jobs_count = 3
  all_machines = range(0, machines_count)
  all_jobs = range(0, jobs_count)
  # Define data.
  machines = [[0, 1, 2],
              [0, 2, 1],
              [1, 2]]

  processing_times = [[3, 2, 2],
                      [2, 1, 4],
                      [4, 3]]
  # Computes horizon.
  horizon = 0
  for job in all_jobs:
    horizon += sum(processing_times[job])

  Task = collections.namedtuple('Task', 'start end interval')
  AssignedTask = collections.namedtuple('AssignedTask', 'start job index')

  # Creates jobs.
  all_tasks = {}
  for job in all_jobs:
    for index in range(0, len(machines[job])):
      start_var = model.NewIntVar(0, horizon, 'start_%i_%i' % (job, index))
      duration = processing_times[job][index]
      end_var = model.NewIntVar(0, horizon, 'end_%i_%i' % (job, index))
      interval_var = model.NewIntervalVar(start_var, duration, end_var,
                                          'interval_%i_%i' % (job, index))
      all_tasks[(job, index)] = Task(start=start_var,
                                     end=end_var,
                                     interval=interval_var)

  # Creates sequence variables and add disjunctive constraints.
  for machine in all_machines:
    intervals = []
    for job in all_jobs:
      for index in range(0, len(machines[job])):
        if machines[job][index] == machine:
          intervals.append(all_tasks[(job, index)].interval)
    model.AddNoOverlap(intervals)

  # Add precedence contraints.
  for job in all_jobs:
    for index in range(0, len(machines[job]) - 1):
      model.Add(all_tasks[(job, index + 1)].start >=
                all_tasks[(job, index)].end)

  # Makespan objective.
  obj_var = model.NewIntVar(0, horizon, 'makespan')
  model.AddMaxEquality(
      obj_var, [all_tasks[(job, len(machines[job]) - 1)].end
                for job in all_jobs])
  model.Minimize(obj_var)

  # Solve model.
  solver = cp_model.CpSolver()
  status = solver.Solve(model)

  if status == cp_model.OPTIMAL:
    # Print out makespan.
    print('Optimal Schedule Length: %i' % solver.ObjectiveValue())
    print()

    # Create one list of assigned tasks per machine.
    assigned_jobs = [[] for _ in range(machines_count)]
    for job in all_jobs:
      for index in range(len(machines[job])):
        machine = machines[job][index]
        assigned_jobs[machine].append(
          AssignedTask(start = solver.Value(all_tasks[(job, index)].start),
                       job = job, index = index))

    disp_col_width = 10
    sol_line = ""
    sol_line_tasks = ""

    print("Optimal Schedule", "\n")

    for machine in all_machines:
      # Sort by starting time.
      assigned_jobs[machine].sort()
      sol_line += "Machine " + str(machine) + ": "
      sol_line_tasks += "Machine " + str(machine) + ": "

      for assigned_task in assigned_jobs[machine]:
        name = 'job_%i_%i' % (assigned_task.job, assigned_task.index)
         # Add spaces to output to align columns.
        sol_line_tasks +=  name + " " * (disp_col_width - len(name))
        start = assigned_task.start
        duration = processing_times[assigned_task.job][assigned_task.index]

        sol_tmp = "[%i,%i]" % (start, start + duration)
        # Add spaces to output to align columns.
        sol_line += sol_tmp + " " * (disp_col_width - len(sol_tmp))

      sol_line += "\n"
      sol_line_tasks += "\n"

    print(sol_line_tasks)
    print("Time Intervals for Tasks\n")
    print(sol_line)



def main():
  MinimalCpSat()
  MinimalCpSatAllSolutions()
  SolvingLinearProblem()
  MinimalJobShop()


if __name__ == '__main__':
  main()
