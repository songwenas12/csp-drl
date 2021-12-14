************************************************************************
file with basedata            : me18_.bas
initial value random generator: 363710587
************************************************************************
projects                      :  1
jobs (incl. supersource/sink ):  16
horizon                       :  107
RESOURCES
  - renewable                 :  2   R
  - nonrenewable              :  0   N
  - doubly constrained        :  0   D
************************************************************************
PROJECT INFORMATION:
pronr.  #jobs rel.date duedate tardcost  MPM-Time
    1     14      0       17        3       17
************************************************************************
PRECEDENCE RELATIONS:
jobnr.    #modes  #successors   successors
   1        1          3           2   3   4
   2        3          2           6   8
   3        3          2          12  13
   4        3          3           5   8  13
   5        3          3          10  11  14
   6        3          3           7   9  10
   7        3          3          11  13  14
   8        3          1          15
   9        3          3          11  12  14
  10        3          1          12
  11        3          1          15
  12        3          1          15
  13        3          1          16
  14        3          1          16
  15        3          1          16
  16        1          0        
************************************************************************
REQUESTS/DURATIONS:
jobnr. mode duration  R 1  R 2
------------------------------------------------------------------------
  1      1     0       0    0
  2      1     6       7    0
         2     7       6    0
         3     9       0    1
  3      1     6       7    0
         2     7       0    5
         3     9       5    0
  4      1     3       6    0
         2     5       0    7
         3     9       5    0
  5      1     4       6    0
         2     5       0    9
         3     9       0    8
  6      1     1       0    4
         2     5       8    0
         3     7       0    1
  7      1     3      10    0
         2     7       0    6
         3     8       6    0
  8      1     2       8    0
         2     4       0   10
         3     5       4    0
  9      1     1       6    0
         2     9       5    0
         3     9       0    7
 10      1     1       0    8
         2     4       7    0
         3     4       0    6
 11      1     2       0    3
         2     3       4    0
         3    10       2    0
 12      1     1       0    4
         2     1       5    0
         3     6       0    3
 13      1     3       4    0
         2     7       0    8
         3     9       0    4
 14      1     7       0    8
         2    10       0    3
         3    10       3    0
 15      1     2       7    0
         2     3       0    8
         3     3       6    0
 16      1     0       0    0
************************************************************************
RESOURCEAVAILABILITIES:
  R 1  R 2
   12   16
************************************************************************
