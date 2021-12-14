************************************************************************
file with basedata            : md121_.bas
initial value random generator: 2010199112
************************************************************************
projects                      :  1
jobs (incl. supersource/sink ):  14
horizon                       :  94
RESOURCES
  - renewable                 :  2   R
  - nonrenewable              :  2   N
  - doubly constrained        :  0   D
************************************************************************
PROJECT INFORMATION:
pronr.  #jobs rel.date duedate tardcost  MPM-Time
    1     12      0       22        8       22
************************************************************************
PRECEDENCE RELATIONS:
jobnr.    #modes  #successors   successors
   1        1          3           2   3   4
   2        3          3           8   9  10
   3        3          3           6   7   9
   4        3          3           5   6   8
   5        3          1           9
   6        3          3          10  11  12
   7        3          2          10  12
   8        3          1          11
   9        3          3          11  12  13
  10        3          1          13
  11        3          1          14
  12        3          1          14
  13        3          1          14
  14        1          0        
************************************************************************
REQUESTS/DURATIONS:
jobnr. mode duration  R 1  R 2  N 1  N 2
------------------------------------------------------------------------
  1      1     0       0    0    0    0
  2      1     1       0    6    8    3
         2     5       4    0    8    3
         3     8       0    4    5    3
  3      1     9       0    6    8    6
         2     9       7    0    9    6
         3     9       0    5    9    4
  4      1     1       0    7    7    5
         2     3       0    6    4    5
         3     9       8    0    3    4
  5      1     2       9    0    3    8
         2     6       0    4    3    6
         3     6       5    0    2    5
  6      1     5       0    2    7   10
         2     8       2    0    7    8
         3    10       2    0    6    6
  7      1     3       0   10    6    7
         2     4       0    9    3    6
         3     8       2    0    1    4
  8      1     4       0    5   10    8
         2     6       8    0    9    7
         3     9       0    3    8    2
  9      1     2       2    0    7   10
         2     4       0    8    4    5
         3     9       0    6    3    5
 10      1     5       0    7   10    7
         2     5       9    0    9    9
         3     9       5    0    9    4
 11      1     4       0    6    5    9
         2     5       0    6    4    8
         3     6       0    5    3    7
 12      1     1       5    0    7    3
         2     2       1    0    5    2
         3     3       0    3    3    1
 13      1     3       0    5   10    6
         2     5       0    5    5    3
         3     8       0    5    2    1
 14      1     0       0    0    0    0
************************************************************************
RESOURCEAVAILABILITIES:
  R 1  R 2  N 1  N 2
    5    9   89   84
************************************************************************
