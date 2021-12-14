************************************************************************
file with basedata            : cr312_.bas
initial value random generator: 1882060696
************************************************************************
projects                      :  1
jobs (incl. supersource/sink ):  18
horizon                       :  116
RESOURCES
  - renewable                 :  3   R
  - nonrenewable              :  2   N
  - doubly constrained        :  0   D
************************************************************************
PROJECT INFORMATION:
pronr.  #jobs rel.date duedate tardcost  MPM-Time
    1     16      0       23        9       23
************************************************************************
PRECEDENCE RELATIONS:
jobnr.    #modes  #successors   successors
   1        1          3           2   3   4
   2        3          3           6  12  14
   3        3          3          10  11  17
   4        3          3           5   7  11
   5        3          3           6   8  12
   6        3          2           9  13
   7        3          2           9  16
   8        3          1          10
   9        3          2          15  17
  10        3          2          14  15
  11        3          1          13
  12        3          3          13  16  17
  13        3          1          15
  14        3          1          16
  15        3          1          18
  16        3          1          18
  17        3          1          18
  18        1          0        
************************************************************************
REQUESTS/DURATIONS:
jobnr. mode duration  R 1  R 2  R 3  N 1  N 2
------------------------------------------------------------------------
  1      1     0       0    0    0    0    0
  2      1     3       6    0    0    0    6
         2     4       3    0    9    0    3
         3     6       3    8    9    0    2
  3      1     5       0    9    2    0    1
         2     7       4    8    0    0    1
         3     8       0    8    0    4    0
  4      1     6       0    7    6    2    0
         2     7       0    0    5    2    0
         3    10       7    0    0    0    3
  5      1     1       6    0    0   10    0
         2     1       8    0    0    9    0
         3     7       4    0    0    5    0
  6      1     9       0    0    6    0    6
         2     9       7    4    0    0    2
         3     9       5    0    6    2    0
  7      1     1       2    0    3    7    0
         2     3       2    0    2    3    0
         3     4       0    0    2    0    3
  8      1     1       0    3    0    6    0
         2     8       0    0    8    6    0
         3    10       0    0    6    5    0
  9      1     1       4    0    0    9    0
         2     4       3    0    0    0    4
         3     7       3    0    8    8    0
 10      1     6       0    0   10    0    3
         2    10       4    0    0    0    3
         3    10       0    6    7    0    2
 11      1     2       0    0    6    3    0
         2     3       0    0    6    0    9
         3     5       6    0    0    3    0
 12      1     2       7    0    0    4    0
         2     9       5    4    7    0    8
         3     9       5    0    7    0    9
 13      1     2       3    5    0    0    7
         2     2       2    0    0    7    0
         3     2       0    0    7    0    8
 14      1     5       0    5   10    0    8
         2     5       9    4    0    0    8
         3     6       9    0    9    0    7
 15      1     5       7    8    9    0    7
         2     5       7    8    0    9    0
         3     7       0    0    8    8    0
 16      1     1       8    8    0    6    0
         2     3       0    0   10    0    9
         3     6       3    0    0    0    9
 17      1     3       0    6   10    8    0
         2     9       0    5    0    0    6
         3    10       3    5    0    8    0
 18      1     0       0    0    0    0    0
************************************************************************
RESOURCEAVAILABILITIES:
  R 1  R 2  R 3  N 1  N 2
   24   24   29   44   47
************************************************************************
