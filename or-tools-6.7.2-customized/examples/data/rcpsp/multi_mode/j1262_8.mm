************************************************************************
file with basedata            : md126_.bas
initial value random generator: 366028125
************************************************************************
projects                      :  1
jobs (incl. supersource/sink ):  14
horizon                       :  96
RESOURCES
  - renewable                 :  2   R
  - nonrenewable              :  2   N
  - doubly constrained        :  0   D
************************************************************************
PROJECT INFORMATION:
pronr.  #jobs rel.date duedate tardcost  MPM-Time
    1     12      0       19        3       19
************************************************************************
PRECEDENCE RELATIONS:
jobnr.    #modes  #successors   successors
   1        1          3           2   3   4
   2        3          2           9  10
   3        3          3           5   6   8
   4        3          3           7   9  10
   5        3          3           7  12  13
   6        3          2          10  11
   7        3          1          11
   8        3          2           9  13
   9        3          2          11  12
  10        3          2          12  13
  11        3          1          14
  12        3          1          14
  13        3          1          14
  14        1          0        
************************************************************************
REQUESTS/DURATIONS:
jobnr. mode duration  R 1  R 2  N 1  N 2
------------------------------------------------------------------------
  1      1     0       0    0    0    0
  2      1     2       4    9    4    6
         2     5       4    6    3    5
         3     8       3    6    2    5
  3      1     6       8    9    5   10
         2     8       8    8    4    7
         3     8       7    8    5    6
  4      1     6       6    8    4    9
         2     7       3    7    3    7
         3     7       3    8    2    5
  5      1     3       5    5    9    7
         2     5       4    5    7    6
         3     9       3    4    3    5
  6      1     4       7    2    8    8
         2     5       6    2    6    6
         3     6       4    2    6    4
  7      1     6       5    4    7    4
         2     6       5    3    6    6
         3     7       3    3    3    4
  8      1     2      10    8    7    5
         2     4       6    6    7    3
         3    10       5    6    7    3
  9      1     4       3    7    9    8
         2     9       2    5    9    6
         3    10       1    3    8    4
 10      1     1       6    6   10    9
         2     7       6    4    5    4
         3    10       3    4    3    1
 11      1     4       4   10    5    2
         2     5       3   10    5    2
         3     8       2   10    4    1
 12      1     5       8    6    9    9
         2     6       8    4    9    5
         3     9       8    3    9    5
 13      1     2       9    5    7    4
         2     2       9    5    9    3
         3     4       9    5    5    3
 14      1     0       0    0    0    0
************************************************************************
RESOURCEAVAILABILITIES:
  R 1  R 2  N 1  N 2
   16   18   86   83
************************************************************************
