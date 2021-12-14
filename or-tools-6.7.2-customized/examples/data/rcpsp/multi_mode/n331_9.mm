************************************************************************
file with basedata            : cn331_.bas
initial value random generator: 1918362155
************************************************************************
projects                      :  1
jobs (incl. supersource/sink ):  18
horizon                       :  138
RESOURCES
  - renewable                 :  2   R
  - nonrenewable              :  3   N
  - doubly constrained        :  0   D
************************************************************************
PROJECT INFORMATION:
pronr.  #jobs rel.date duedate tardcost  MPM-Time
    1     16      0       29        3       29
************************************************************************
PRECEDENCE RELATIONS:
jobnr.    #modes  #successors   successors
   1        1          3           2   3   4
   2        3          3           9  11  13
   3        3          3           5   6  12
   4        3          1          14
   5        3          2           7   8
   6        3          2           9  16
   7        3          3          10  13  16
   8        3          3           9  13  16
   9        3          1          10
  10        3          2          15  17
  11        3          3          12  14  15
  12        3          1          17
  13        3          2          14  15
  14        3          1          17
  15        3          1          18
  16        3          1          18
  17        3          1          18
  18        1          0        
************************************************************************
REQUESTS/DURATIONS:
jobnr. mode duration  R 1  R 2  N 1  N 2  N 3
------------------------------------------------------------------------
  1      1     0       0    0    0    0    0
  2      1     2       9   10    0    9    0
         2     3       8    7    6    4    0
         3    10       8    2    3    0    9
  3      1     1       8    9    4   10    1
         2     5       7    9    4    8    0
         3     9       3    8    0    7    0
  4      1     1       4    6    3    9    0
         2     4       3    3    0    0    8
         3     8       3    1    0    6    0
  5      1     5       6    6    8    8    9
         2     7       4    6    4    8    0
         3     9       3    5    0    0    9
  6      1     2       3    9    9    0   10
         2     4       3    5    7    6    0
         3     6       3    5    6    0    0
  7      1     7       7    5    3    0    0
         2     8       4    4    0    4    6
         3     9       3    1    0    0    6
  8      1     6       8    7    0    8    6
         2     9       8    6    0    6    0
         3    10       8    3    9    0    0
  9      1     4       3    5    6    0    0
         2     5       3    3    5    7    4
         3     9       2    1    0    7    0
 10      1     5       8    6    0    6    0
         2     7       6    5    1    3    0
         3    10       5    3    0    0    7
 11      1     1       5    5    0    2    0
         2     1       2    2    0    0    2
         3     1       5    3    0    3    0
 12      1     6       9    9    9    0    6
         2     7       4    7    9    0    5
         3     8       1    6    0    6    5
 13      1     3       7    5    0    0    8
         2     6       7    4    0    0    3
         3    10       4    3    5    0    0
 14      1     4       4    8    0   10    5
         2     4       4    6    6    0    5
         3     9       4    6    0    0    3
 15      1     7       7    5    0    8    0
         2     8       7    3    8    8    5
         3    10       6    2    5    0    0
 16      1     3       8   10    7    0    5
         2     6       4    9    0    3    0
         3    10       4    7    4    0    0
 17      1     8       8    8    8    0    0
         2     9       6    4    2    0    8
         3    10       4    1    0    0    6
 18      1     0       0    0    0    0    0
************************************************************************
RESOURCEAVAILABILITIES:
  R 1  R 2  N 1  N 2  N 3
   20   21   92   97   99
************************************************************************
