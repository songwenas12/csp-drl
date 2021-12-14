************************************************************************
file with basedata            : md353_.bas
initial value random generator: 2016537265
************************************************************************
projects                      :  1
jobs (incl. supersource/sink ):  22
horizon                       :  142
RESOURCES
  - renewable                 :  2   R
  - nonrenewable              :  2   N
  - doubly constrained        :  0   D
************************************************************************
PROJECT INFORMATION:
pronr.  #jobs rel.date duedate tardcost  MPM-Time
    1     20      0       23       16       23
************************************************************************
PRECEDENCE RELATIONS:
jobnr.    #modes  #successors   successors
   1        1          3           2   3   4
   2        3          3           5   6  18
   3        3          3          10  17  19
   4        3          3           7  12  17
   5        3          3           9  11  12
   6        3          3           7  14  21
   7        3          2           8  11
   8        3          2           9  16
   9        3          2          19  20
  10        3          1          13
  11        3          1          15
  12        3          3          14  15  21
  13        3          2          18  20
  14        3          2          16  20
  15        3          1          16
  16        3          1          19
  17        3          1          18
  18        3          1          21
  19        3          1          22
  20        3          1          22
  21        3          1          22
  22        1          0        
************************************************************************
REQUESTS/DURATIONS:
jobnr. mode duration  R 1  R 2  N 1  N 2
------------------------------------------------------------------------
  1      1     0       0    0    0    0
  2      1     4       0    7   10    8
         2     5       0    5    6    8
         3    10       2    0    4    8
  3      1     3       4    0    7    6
         2     4       4    0    5    5
         3     4       0    7    6    4
  4      1     2       9    0    7    6
         2     4       4    0    3    6
         3     4       0    7    3    2
  5      1     1       0    7    5    8
         2     2       2    0    5    3
         3     9       0    7    2    2
  6      1     1       3    0    8    9
         2     2       1    0    7    7
         3     4       0    2    7    5
  7      1     4       9    0   10    8
         2     6       0    4    9    7
         3     8       0    4    7    6
  8      1     1       0    8    8    8
         2     4       8    0    8    8
         3     7       0    8    6    8
  9      1     5       5    0    3    3
         2     6       0    7    3    2
         3     7       0    6    2    1
 10      1     8       8    0    4    7
         2     9       0   10    3    6
         3     9       8    0    2    7
 11      1     1       9    0    8    8
         2     7       7    0    6    6
         3     9       0   10    5    3
 12      1     4       0   10    7    2
         2     4       1    0    8    2
         3     6       0   10    6    2
 13      1     2       0    7    7    7
         2     9       7    0    7    6
         3     9       0    4    7    4
 14      1     6       5    0    5    6
         2    10       0    2    5    4
         3    10       3    0    4    1
 15      1     3       0    7    8    4
         2     5       0    6    4    3
         3     6       0    5    1    3
 16      1     2       8    0    4    8
         2     4       0   10    4    8
         3     7       0    7    4    7
 17      1     5       0    2    8    8
         2     5       6    0    7    5
         3     7       5    0    6    4
 18      1     2       0    8    6    5
         2     6       0    8    3    5
         3     8       7    0    2    4
 19      1     6       4    0    4    4
         2     6       5    0    5    3
         3     6       0    3    5    2
 20      1     3       7    0    9   10
         2     6       0    8    9    7
         3     7       5    0    8    7
 21      1     2       0    3    6    9
         2     2       2    0    8    9
         3     5       0    3    6    8
 22      1     0       0    0    0    0
************************************************************************
RESOURCEAVAILABILITIES:
  R 1  R 2  N 1  N 2
    6   11  103   99
************************************************************************
