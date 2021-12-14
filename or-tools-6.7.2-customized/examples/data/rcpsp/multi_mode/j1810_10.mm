************************************************************************
file with basedata            : md266_.bas
initial value random generator: 82389093
************************************************************************
projects                      :  1
jobs (incl. supersource/sink ):  20
horizon                       :  150
RESOURCES
  - renewable                 :  2   R
  - nonrenewable              :  2   N
  - doubly constrained        :  0   D
************************************************************************
PROJECT INFORMATION:
pronr.  #jobs rel.date duedate tardcost  MPM-Time
    1     18      0       30        2       30
************************************************************************
PRECEDENCE RELATIONS:
jobnr.    #modes  #successors   successors
   1        1          3           2   3   4
   2        3          1          11
   3        3          3           5  10  13
   4        3          2           6   7
   5        3          2          14  19
   6        3          3           8   9  11
   7        3          3           8   9  16
   8        3          1          12
   9        3          1          15
  10        3          2          11  16
  11        3          2          12  14
  12        3          1          15
  13        3          2          15  16
  14        3          2          17  18
  15        3          3          17  18  19
  16        3          2          18  19
  17        3          1          20
  18        3          1          20
  19        3          1          20
  20        1          0        
************************************************************************
REQUESTS/DURATIONS:
jobnr. mode duration  R 1  R 2  N 1  N 2
------------------------------------------------------------------------
  1      1     0       0    0    0    0
  2      1     3       3    0   10    0
         2     8       3    0    9    0
         3    10       0    9    0    2
  3      1     2       0    9    0    7
         2     8       0    4    1    0
         3     9       7    0    0    4
  4      1     1       4    0   10    0
         2     5       0    5    7    0
         3     6       3    0    0    6
  5      1     4       0    7   10    0
         2     8       0    7    6    0
         3     9       0    5    0    7
  6      1     1       6    0    3    0
         2     7       5    0    0    2
         3     9       5    0    3    0
  7      1     1       0    7   10    0
         2     2      10    0    8    0
         3     7       0    7    0    7
  8      1     2       0    5    0    3
         2     2       0    6    6    0
         3     3       2    0    2    0
  9      1     2       6    0    6    0
         2     9       6    0    0    6
         3     9       0    8    0    6
 10      1     8       5    0   10    0
         2     8       0    4    8    0
         3    10       6    0    4    0
 11      1     5       0    4    0    6
         2    10       0    3    8    0
         3    10       3    0    8    0
 12      1     9       6    0    3    0
         2     9       0    8    4    0
         3     9       3    0    0    7
 13      1     3       5    0    5    0
         2     9       0    3    0    7
         3    10       0    3    0    6
 14      1     3       0    7    0    6
         2     3       6    0    7    0
         3    10       3    0    0    7
 15      1     1       0    5    6    0
         2     7       6    0    0    7
         3     8       0    4    0    3
 16      1     1       4    0    7    0
         2     7       3    0    0   10
         3     8       1    0    0    7
 17      1     5       0    8    9    0
         2    10       0    7    8    0
         3    10       3    0    7    0
 18      1     2       0    5    4    0
         2     3       4    0    4    0
         3     3       3    0    0   10
 19      1     3       4    0    0    6
         2    10       0    6    9    0
         3    10       4    0    9    0
 20      1     0       0    0    0    0
************************************************************************
RESOURCEAVAILABILITIES:
  R 1  R 2  N 1  N 2
   16   18   68   50
************************************************************************
