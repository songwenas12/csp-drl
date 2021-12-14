************************************************************************
file with basedata            : cr324_.bas
initial value random generator: 24634
************************************************************************
projects                      :  1
jobs (incl. supersource/sink ):  18
horizon                       :  129
RESOURCES
  - renewable                 :  3   R
  - nonrenewable              :  2   N
  - doubly constrained        :  0   D
************************************************************************
PROJECT INFORMATION:
pronr.  #jobs rel.date duedate tardcost  MPM-Time
    1     16      0       16        9       16
************************************************************************
PRECEDENCE RELATIONS:
jobnr.    #modes  #successors   successors
   1        1          3           2   3   4
   2        3          3           7   8   9
   3        3          2           5  11
   4        3          3          12  13  14
   5        3          3           6   7  14
   6        3          2          12  13
   7        3          2          16  17
   8        3          3          10  15  17
   9        3          2          11  13
  10        3          2          11  12
  11        3          1          16
  12        3          1          16
  13        3          2          15  17
  14        3          1          15
  15        3          1          18
  16        3          1          18
  17        3          1          18
  18        1          0        
************************************************************************
REQUESTS/DURATIONS:
jobnr. mode duration  R 1  R 2  R 3  N 1  N 2
------------------------------------------------------------------------
  1      1     0       0    0    0    0    0
  2      1     2       9    8    1    7    0
         2     3       9    8    1    6    0
         3     8       9    7    1    0   10
  3      1     1       3    5    5    0    5
         2     4       3    4    5    0    5
         3     7       3    2    4    6    0
  4      1     1       8    9    5    8    0
         2     7       4    9    4    8    0
         3     7       7    8    1    0    4
  5      1     4       8    7    6    0    6
         2     6       7    7    5    0    5
         3    10       6    7    5    3    0
  6      1     4       3    3    5    0    4
         2     4       2    3    6    7    0
         3     5       2    2    5    0    5
  7      1     2       4    6    8    9    0
         2     7       4    5    6    0    8
         3     8       4    5    6    0    6
  8      1     3       7   10    8   10    0
         2     3       4    8   10    0    5
         3    10       4    3    7   10    0
  9      1     5       6    9    7    0    9
         2     5       6    9    9    0    8
         3     8       5    8    7    0    8
 10      1     2       6    8    7    9    0
         2     6       4    7    5    0   10
         3     8       1    5    4    0   10
 11      1     2       7    4    8    5    0
         2     6       7    1    3    5    0
         3     6       5    1    5    2    0
 12      1     6       9   10    9    0    7
         2     9       7    6    6    0    5
         3    10       6    6    4    9    0
 13      1     1       5    7    4    0    7
         2     4       4    6    4    0    7
         3     5       3    5    4    0    6
 14      1     8       6    2    1    0    8
         2     9       6    1    1    3    0
         3     9       6    1    1    0    6
 15      1     3       6   10    4    3    0
         2     5       5    8    3    0    5
         3    10       5    8    2    3    0
 16      1     1       4    9    3    6    0
         2     7       4    6    3    5    0
         3    10       3    6    2    3    0
 17      1     2       6    7    6    0    8
         2     3       5    5    6    9    0
         3     8       4    4    6    0    5
 18      1     0       0    0    0    0    0
************************************************************************
RESOURCEAVAILABILITIES:
  R 1  R 2  R 3  N 1  N 2
   25   28   31   72   76
************************************************************************
