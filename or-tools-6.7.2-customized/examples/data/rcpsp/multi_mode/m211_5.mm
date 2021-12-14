************************************************************************
file with basedata            : cm211_.bas
initial value random generator: 21871247
************************************************************************
projects                      :  1
jobs (incl. supersource/sink ):  18
horizon                       :  127
RESOURCES
  - renewable                 :  2   R
  - nonrenewable              :  2   N
  - doubly constrained        :  0   D
************************************************************************
PROJECT INFORMATION:
pronr.  #jobs rel.date duedate tardcost  MPM-Time
    1     16      0       27       10       27
************************************************************************
PRECEDENCE RELATIONS:
jobnr.    #modes  #successors   successors
   1        1          3           2   3   4
   2        2          3          11  14  17
   3        2          3           6   9  10
   4        2          3           5   8  12
   5        2          2           9  13
   6        2          3           7  13  15
   7        2          1          12
   8        2          3          10  11  17
   9        2          2          14  16
  10        2          2          14  16
  11        2          2          13  15
  12        2          1          17
  13        2          1          16
  14        2          1          15
  15        2          1          18
  16        2          1          18
  17        2          1          18
  18        1          0        
************************************************************************
REQUESTS/DURATIONS:
jobnr. mode duration  R 1  R 2  N 1  N 2
------------------------------------------------------------------------
  1      1     0       0    0    0    0
  2      1     5       5    0    0    2
         2    10       3    0    9    0
  3      1     7       0    7    7    0
         2     8       0    7    3    0
  4      1     2       2    0    0    8
         2     7       2    0    8    0
  5      1    10       6    0    8    0
         2    10       0    2    0   10
  6      1     1       0   10    0    8
         2     9       0    9    7    0
  7      1     4       9    0    1    0
         2     7       6    0    1    0
  8      1     5       0    7    0    6
         2     9       0    2    5    0
  9      1     7       0    6    8    0
         2     7       7    0    8    0
 10      1     3       0    2    0    4
         2     6       1    0    5    0
 11      1     8       7    0    9    0
         2     9       4    0    9    0
 12      1     5       0    9    3    0
         2     6       3    0    3    0
 13      1     5       0    6    0    6
         2    10       5    0    9    0
 14      1     6       8    0    8    0
         2     7       0    5    8    0
 15      1     2       0    5    6    0
         2    10       3    0    1    0
 16      1     7       3    0    0    7
         2     7       2    0    8    0
 17      1     4       0    6    0    7
         2     5       2    0    0    4
 18      1     0       0    0    0    0
************************************************************************
RESOURCEAVAILABILITIES:
  R 1  R 2  N 1  N 2
   19   18   67   31
************************************************************************
