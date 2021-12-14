************************************************************************
file with basedata            : cm558_.bas
initial value random generator: 749189567
************************************************************************
projects                      :  1
jobs (incl. supersource/sink ):  18
horizon                       :  143
RESOURCES
  - renewable                 :  2   R
  - nonrenewable              :  2   N
  - doubly constrained        :  0   D
************************************************************************
PROJECT INFORMATION:
pronr.  #jobs rel.date duedate tardcost  MPM-Time
    1     16      0       14        2       14
************************************************************************
PRECEDENCE RELATIONS:
jobnr.    #modes  #successors   successors
   1        1          3           2   3   4
   2        5          3           5   6   7
   3        5          3           9  11  15
   4        5          3          12  13  17
   5        5          2           8  10
   6        5          3           8  10  13
   7        5          3           8  12  13
   8        5          1          11
   9        5          2          16  17
  10        5          2          14  17
  11        5          1          16
  12        5          1          15
  13        5          1          14
  14        5          2          15  16
  15        5          1          18
  16        5          1          18
  17        5          1          18
  18        1          0        
************************************************************************
REQUESTS/DURATIONS:
jobnr. mode duration  R 1  R 2  N 1  N 2
------------------------------------------------------------------------
  1      1     0       0    0    0    0
  2      1     2       5    0    3    4
         2     3       4    0    2    4
         3     4       0    7    2    3
         4     9       0    6    2    2
         5    10       2    0    1    2
  3      1     3       0    8    7    1
         2     5       0    3    7    1
         3     5       6    0    7    1
         4     6       0    3    6    1
         5     8       6    0    6    1
  4      1     1       0    8    6    6
         2     1       6    0    6    7
         3     5       5    0    6    6
         4     7       5    0    5    6
         5    10       3    0    5    5
  5      1     1       0    9    4    7
         2     3       0    8    4    7
         3     7       0    7    3    7
         4     8       0    6    2    7
         5     9       0    5    1    7
  6      1     4       7    0    2    5
         2     4       0    6    2    6
         3     5       6    0    2    4
         4     7       6    0    2    3
         5     8       4    0    1    2
  7      1     5       0    3    2    4
         2     5       8    0    2    4
         3     6       0    3    2    3
         4     8       6    0    2    3
         5     8       0    2    2    3
  8      1     2       0    6   10    7
         2     6       9    0    9    7
         3     9       0    6    6    4
         4    10       0    5    6    1
         5    10       9    0    6    2
  9      1     1       5    0    9    9
         2     1       0    8    8    9
         3     2       5    0    8    9
         4     7       3    0    6    8
         5     8       2    0    6    8
 10      1     4       0    8    7   10
         2     4       6    0    7   10
         3     6       0    7    7    8
         4     7       0    7    7    7
         5    10       3    0    7    6
 11      1     1       0    4    3   10
         2     1       4    0    4   10
         3     7       0    5    2   10
         4     7       0    5    3    9
         5     8       4    0    1    9
 12      1     1       0    5    9    5
         2     3      10    0    8    5
         3     6      10    0    5    4
         4     6       9    0    6    3
         5     8       0    4    3    2
 13      1     1       4    0    5    5
         2     2       0    7    5    5
         3     5       0    7    3    4
         4     6       4    0    2    4
         5     7       2    0    2    3
 14      1     3       0   10   10    7
         2     5       0    8    8    6
         3     8       0    8    8    5
         4     9       0    6    6    5
         5    10       5    0    6    5
 15      1     1       4    0   10    6
         2     1       5    0   10    5
         3     6       0   10    9    5
         4     7       0    9    9    3
         5    10       3    0    8    2
 16      1     1       7    0    7   10
         2     7       0    5    7   10
         3     8       0    4    6    9
         4     8       5    0    7    9
         5    10       4    0    6    8
 17      1     2       8    0    9    9
         2     6       0    8    7    8
         3     6       7    0    7    9
         4     7       0    8    6    8
         5     9       0    8    4    7
 18      1     0       0    0    0    0
************************************************************************
RESOURCEAVAILABILITIES:
  R 1  R 2  N 1  N 2
   15   16  104  107
************************************************************************
