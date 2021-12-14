************************************************************************
file with basedata            : md152_.bas
initial value random generator: 192700689
************************************************************************
projects                      :  1
jobs (incl. supersource/sink ):  16
horizon                       :  110
RESOURCES
  - renewable                 :  2   R
  - nonrenewable              :  2   N
  - doubly constrained        :  0   D
************************************************************************
PROJECT INFORMATION:
pronr.  #jobs rel.date duedate tardcost  MPM-Time
    1     14      0       23        2       23
************************************************************************
PRECEDENCE RELATIONS:
jobnr.    #modes  #successors   successors
   1        1          3           2   3   4
   2        3          3           9  10  13
   3        3          3           8  10  11
   4        3          3           5   6   8
   5        3          1          11
   6        3          1           7
   7        3          3           9  10  11
   8        3          2           9  12
   9        3          1          14
  10        3          2          14  15
  11        3          2          13  15
  12        3          2          13  15
  13        3          1          16
  14        3          1          16
  15        3          1          16
  16        1          0        
************************************************************************
REQUESTS/DURATIONS:
jobnr. mode duration  R 1  R 2  N 1  N 2
------------------------------------------------------------------------
  1      1     0       0    0    0    0
  2      1     1       9    5    4    0
         2     5       9    5    0    7
         3     7       9    4    3    0
  3      1     6       3    7   10    0
         2     8       1    5    7    0
         3     8       2    6    0    7
  4      1     3       9    6    0    6
         2     8       9    2    0    6
         3     8       9    4    0    5
  5      1     3       9    6    7    0
         2     9       7    5    5    0
         3    10       5    4    0    3
  6      1     5       8    3    2    0
         2     7       6    3    0    5
         3     9       3    3    0    4
  7      1     6       7    6    0   10
         2     7       5    6    0   10
         3     7       5    5    4    0
  8      1     5      10    9    4    0
         2     5      10    7    6    0
         3     8       9    6    0    8
  9      1     6       8    6    7    0
         2     9       6    4    3    0
         3    10       2    3    0    4
 10      1     2       7    8    4    0
         2     5       7    7    4    0
         3     6       7    7    0    6
 11      1     3       3   10    0    5
         2     5       3    9    6    0
         3     8       3    7    5    0
 12      1     3       8    6    5    0
         2     5       6    5    4    0
         3     6       5    3    3    0
 13      1     2       5    6    0   10
         2     3       4    4    0    8
         3     6       3    2    8    0
 14      1     3       4    9    9    0
         2    10       4    8    6    0
         3    10       4    7    0    4
 15      1     2       6    6    5    0
         2     5       4    6    0    4
         3     7       4    5    0    3
 16      1     0       0    0    0    0
************************************************************************
RESOURCEAVAILABILITIES:
  R 1  R 2  N 1  N 2
   21   24   59   61
************************************************************************
