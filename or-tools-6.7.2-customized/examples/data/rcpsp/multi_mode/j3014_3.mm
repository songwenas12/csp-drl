************************************************************************
file with basedata            : mf14_.bas
initial value random generator: 707646053
************************************************************************
projects                      :  1
jobs (incl. supersource/sink ):  32
horizon                       :  246
RESOURCES
  - renewable                 :  2   R
  - nonrenewable              :  2   N
  - doubly constrained        :  0   D
************************************************************************
PROJECT INFORMATION:
pronr.  #jobs rel.date duedate tardcost  MPM-Time
    1     30      0       29       26       29
************************************************************************
PRECEDENCE RELATIONS:
jobnr.    #modes  #successors   successors
   1        1          3           2   3   4
   2        3          2           7  18
   3        3          3          13  22  27
   4        3          3           5   6   8
   5        3          2          11  15
   6        3          3           9  15  20
   7        3          3          11  12  22
   8        3          2          14  16
   9        3          1          10
  10        3          2          22  23
  11        3          1          28
  12        3          1          14
  13        3          3          17  23  25
  14        3          2          20  27
  15        3          3          16  17  29
  16        3          2          19  28
  17        3          2          19  21
  18        3          3          21  24  26
  19        3          1          30
  20        3          1          21
  21        3          2          30  31
  22        3          3          25  26  31
  23        3          1          29
  24        3          2          25  27
  25        3          1          29
  26        3          1          28
  27        3          1          31
  28        3          1          30
  29        3          1          32
  30        3          1          32
  31        3          1          32
  32        1          0        
************************************************************************
REQUESTS/DURATIONS:
jobnr. mode duration  R 1  R 2  N 1  N 2
------------------------------------------------------------------------
  1      1     0       0    0    0    0
  2      1     5       9    8    0    4
         2     5       6    8   10    0
         3     6       6    5    0    5
  3      1     4       9    9    0    7
         2     6       8    9    6    0
         3     6       9    9    0    4
  4      1     1       4    8    5    0
         2     1       6    9    0    6
         3     7       1    6    0    6
  5      1     1       7    6    6    0
         2     3       4    5    0    9
         3     6       2    5    0    9
  6      1     1       5    4    0    4
         2     3       4    4    8    0
         3     7       4    3    7    0
  7      1     6      10   10   10    0
         2     7       6    9    9    0
         3    10       4    9    8    0
  8      1     1       8    5    5    0
         2     4       8    5    4    0
         3     8       6    3    0    7
  9      1     6       6   10    2    0
         2     6       8   10    0    3
         3     9       4    9    0    3
 10      1     4       4    8    0    7
         2     4       3    8    0    9
         3    10       1    7    0    5
 11      1     2       5   10    8    0
         2     2       5    9    0    8
         3     9       5    8    0    7
 12      1     6       3    9    0    7
         2     6       4    9    3    0
         3     7       2    9    0    8
 13      1     5       5    9    0    9
         2     5       5    9    6    0
         3     9       4    8    5    0
 14      1     2       7    5    8    0
         2     2      10    5    0    8
         3     9       4    5    0    8
 15      1     2       8    8    7    0
         2     2       9    6    6    0
         3     4       7    6    6    0
 16      1     3      10    6    0    6
         2     5       8    3    1    0
         3     6       5    2    0    5
 17      1     7       7    8    0    5
         2    10       5    4    2    0
         3    10       2    6    2    0
 18      1     4      10   10    2    0
         2     9       5    9    0    4
         3    10       4    9    0    3
 19      1     3       9    6    0    7
         2     4       5    6    0    6
         3     9       4    4    7    0
 20      1     2       8    3    9    0
         2     3       8    3    0    4
         3     6       8    3    5    0
 21      1     6       8    4    0    6
         2     9       6    3    3    0
         3     9       6    2    0    5
 22      1     3       9    8    3    0
         2     5       9    5    2    0
         3     8       7    3    2    0
 23      1     6       8    7    5    0
         2     8       8    7    0    5
         3     9       6    2    1    0
 24      1     6       3    6    9    0
         2     8       3    4    6    0
         3    10       2    4    0    2
 25      1     2       9    2    0    5
         2     6       9    1    8    0
         3    10       8    1    5    0
 26      1     1       5   10    0    7
         2     6       4    8    0    6
         3     9       2    4    0    6
 27      1     7       4    6    2    0
         2     8       4    3    1    0
         3     9       3    2    0    3
 28      1     3       4    6    9    0
         2    10       4    6    0    4
         3    10       4    6    6    0
 29      1     3       9    9    0    7
         2     8       9    4    6    0
         3     8       9    6    4    0
 30      1     1       4    3    0    9
         2     6       4    3    0    1
         3    10       4    2    3    0
 31      1     2       8    4    3    0
         2     3       4    4    1    0
         3     6       3    3    0   10
 32      1     0       0    0    0    0
************************************************************************
RESOURCEAVAILABILITIES:
  R 1  R 2  N 1  N 2
   26   29   86   89
************************************************************************
