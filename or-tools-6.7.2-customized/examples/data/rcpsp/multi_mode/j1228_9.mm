************************************************************************
file with basedata            : md92_.bas
initial value random generator: 576918701
************************************************************************
projects                      :  1
jobs (incl. supersource/sink ):  14
horizon                       :  104
RESOURCES
  - renewable                 :  2   R
  - nonrenewable              :  2   N
  - doubly constrained        :  0   D
************************************************************************
PROJECT INFORMATION:
pronr.  #jobs rel.date duedate tardcost  MPM-Time
    1     12      0       15       11       15
************************************************************************
PRECEDENCE RELATIONS:
jobnr.    #modes  #successors   successors
   1        1          3           2   3   4
   2        3          3           5   7  12
   3        3          3           5   6   8
   4        3          3           5   7   9
   5        3          1          11
   6        3          2           9  10
   7        3          2           8  10
   8        3          2          11  13
   9        3          2          12  13
  10        3          2          11  13
  11        3          1          14
  12        3          1          14
  13        3          1          14
  14        1          0        
************************************************************************
REQUESTS/DURATIONS:
jobnr. mode duration  R 1  R 2  N 1  N 2
------------------------------------------------------------------------
  1      1     0       0    0    0    0
  2      1     1       8    0    0    7
         2     4       5    0    0    6
         3     7       4    0    9    0
  3      1     1       6    0    0    7
         2     1       0    9    0    3
         3     7       7    0    2    0
  4      1     5       0    6    0    8
         2     6       3    0    7    0
         3     9       0    2    0    8
  5      1     1       3    0    0    1
         2     1       0    3    3    0
         3     9       0    3    2    0
  6      1     4       6    0    8    0
         2     9       0    3    0    4
         3    10       4    0    6    0
  7      1     3       0    7    2    0
         2     6       4    0    0    8
         3    10       0    7    0    4
  8      1     2       6    0    0    6
         2     6       4    0    0    5
         3    10       2    0    6    0
  9      1     4       4    0    8    0
         2     8       0   10    7    0
         3    10       0    6    0    9
 10      1     4       0    7    6    0
         2     8       0    6    0   10
         3    10       0    5    0    8
 11      1     2       0    5    6    0
         2     8       3    0    0    5
         3     9       0    5    0    4
 12      1     3       0    9    4    0
         2     4       4    0    0    5
         3     8       0    5    0    5
 13      1     3       2    0    0    8
         2     3       0    4    0    7
         3     5       0    3    0    6
 14      1     0       0    0    0    0
************************************************************************
RESOURCEAVAILABILITIES:
  R 1  R 2  N 1  N 2
   18   17   61   78
************************************************************************
