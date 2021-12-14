************************************************************************
file with basedata            : md265_.bas
initial value random generator: 1001122077
************************************************************************
projects                      :  1
jobs (incl. supersource/sink ):  20
horizon                       :  144
RESOURCES
  - renewable                 :  2   R
  - nonrenewable              :  2   N
  - doubly constrained        :  0   D
************************************************************************
PROJECT INFORMATION:
pronr.  #jobs rel.date duedate tardcost  MPM-Time
    1     18      0       28        9       28
************************************************************************
PRECEDENCE RELATIONS:
jobnr.    #modes  #successors   successors
   1        1          3           2   3   4
   2        3          3           5   6   7
   3        3          3           8  10  15
   4        3          1          14
   5        3          3           8  10  14
   6        3          3          12  15  17
   7        3          3           8  14  18
   8        3          3           9  11  12
   9        3          1          13
  10        3          3          11  18  19
  11        3          1          13
  12        3          1          19
  13        3          1          16
  14        3          2          16  19
  15        3          1          18
  16        3          1          17
  17        3          1          20
  18        3          1          20
  19        3          1          20
  20        1          0        
************************************************************************
REQUESTS/DURATIONS:
jobnr. mode duration  R 1  R 2  N 1  N 2
------------------------------------------------------------------------
  1      1     0       0    0    0    0
  2      1     4       0    4    0    4
         2     4       5    0    5    0
         3     8       3    0    2    0
  3      1     1       0    9    9    0
         2     3       0    9    0    4
         3     3       0    9    8    0
  4      1     2      10    0    0    6
         2     4       0    6    6    0
         3     8       0    2    0    6
  5      1     3       4    0    4    0
         2     7       4    0    2    0
         3     9       0    1    0    6
  6      1     2       7    0    0    9
         2     3       0    5    0    5
         3    10       6    0    0    1
  7      1     2       9    0    0    8
         2     3       7    0    0    4
         3     8       0    5    0    3
  8      1     2       0    3    7    0
         2     6       6    0    3    0
         3     6       0    3    0    4
  9      1     1       0    4    0    6
         2     3       8    0    0    5
         3     8       6    0    6    0
 10      1     3       2    0    0    1
         2     3       0    8    9    0
         3     9       0    5    5    0
 11      1     2       5    0    0    6
         2     3       2    0    5    0
         3     8       0    7    5    0
 12      1     1       0    6    9    0
         2     4       7    0    9    0
         3     9       5    0    7    0
 13      1     9       4    0    6    0
         2    10       0    3    2    0
         3    10       3    0    0    9
 14      1     1       0    5    5    0
         2     4       0    4    0   10
         3    10       9    0    0    2
 15      1     3       9    0    2    0
         2     4       2    0    0    7
         3     8       0    2    1    0
 16      1     5       0    9    8    0
         2     7       4    0    0    3
         3     9       0    5    0    3
 17      1     2      10    0    0    4
         2     4      10    0    7    0
         3     5      10    0    0    2
 18      1     1       0    7    0    1
         2     5       0    6    5    0
         3     6       0    6    3    0
 19      1     8       0    5   10    0
         2     8       3    0    9    0
         3    10       0    5    7    0
 20      1     0       0    0    0    0
************************************************************************
RESOURCEAVAILABILITIES:
  R 1  R 2  N 1  N 2
   15   12   59   46
************************************************************************
