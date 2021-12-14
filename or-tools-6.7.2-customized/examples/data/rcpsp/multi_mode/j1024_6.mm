************************************************************************
file with basedata            : mm24_.bas
initial value random generator: 290065662
************************************************************************
projects                      :  1
jobs (incl. supersource/sink ):  12
horizon                       :  81
RESOURCES
  - renewable                 :  2   R
  - nonrenewable              :  2   N
  - doubly constrained        :  0   D
************************************************************************
PROJECT INFORMATION:
pronr.  #jobs rel.date duedate tardcost  MPM-Time
    1     10      0       16        9       16
************************************************************************
PRECEDENCE RELATIONS:
jobnr.    #modes  #successors   successors
   1        1          3           2   3   4
   2        3          3           5   7  11
   3        3          1           6
   4        3          1           7
   5        3          2           9  10
   6        3          2           8  10
   7        3          1          10
   8        3          2           9  11
   9        3          1          12
  10        3          1          12
  11        3          1          12
  12        1          0        
************************************************************************
REQUESTS/DURATIONS:
jobnr. mode duration  R 1  R 2  N 1  N 2
------------------------------------------------------------------------
  1      1     0       0    0    0    0
  2      1     6       6    0    9    7
         2     7       0    2    6    6
         3     7       5    0    6    6
  3      1     2       0    3    5    6
         2     3       7    0    5    5
         3     9       0    3    3    4
  4      1     3       6    0    4    5
         2     4       0    4    2    5
         3     5       3    0    2    4
  5      1     5      10    0    3    8
         2     6       0    4    3    6
         3     9       2    0    3    4
  6      1     3       0    2    9    5
         2     7       0    2    9    4
         3     8       0    1    8    4
  7      1     7       7    0   10    7
         2     9       0    5    8    6
         3    10       6    0    8    5
  8      1     4       0    9    6    5
         2     6       5    0    5    2
         3    10       4    0    3    1
  9      1     1      10    0    6    3
         2     1       0    1    5    3
         3     4      10    0    3    2
 10      1     3       2    0    8    4
         2     4       0    3    5    3
         3     9       0    3    3    3
 11      1     7       3    0    4    8
         2     8       2    0    4    7
         3    10       0    8    4    7
 12      1     0       0    0    0    0
************************************************************************
RESOURCEAVAILABILITIES:
  R 1  R 2  N 1  N 2
   15   13   64   58
************************************************************************
