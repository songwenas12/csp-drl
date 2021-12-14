jobs  (incl. supersource/sink ):	52
RESOURCES
- renewable                 : 2 R
- nonrenewable              : 4 N
- doubly constrained        : 0 D
************************************************************************
PRECEDENCE RELATIONS:
jobnr.    #modes  #successors   successors
1	1	6		2 4 5 7 8 12 
2	6	1		3 
3	6	3		14 9 6 
4	6	2		14 6 
5	6	1		6 
6	6	3		15 11 10 
7	6	2		14 9 
8	6	2		15 10 
9	6	2		15 11 
10	6	4		19 18 16 13 
11	6	7		29 25 24 21 19 17 16 
12	6	4		24 20 17 15 
13	6	10		32 29 28 27 25 24 23 22 21 20 
14	6	8		32 28 27 23 22 21 20 18 
15	6	6		28 26 23 22 21 18 
16	6	5		34 32 30 28 23 
17	6	3		27 26 22 
18	6	4		34 33 29 25 
19	6	3		32 28 27 
20	6	4		36 33 30 26 
21	6	5		36 34 33 31 30 
22	6	4		36 33 31 30 
23	6	4		43 36 33 31 
24	6	4		43 36 33 31 
25	6	3		36 31 30 
26	6	3		38 34 31 
27	6	3		38 34 31 
28	6	4		43 39 37 33 
29	6	5		42 39 38 37 35 
30	6	6		51 43 42 39 38 37 
31	6	4		42 39 37 35 
32	6	3		41 37 36 
33	6	5		50 42 41 40 38 
34	6	5		51 49 43 42 39 
35	6	5		50 49 47 41 40 
36	6	4		51 49 42 39 
37	6	4		50 49 47 40 
38	6	5		49 48 47 46 44 
39	6	4		50 47 46 44 
40	6	3		48 46 44 
41	6	3		51 48 46 
42	6	2		47 45 
43	6	2		48 45 
44	6	1		45 
45	6	1		52 
46	6	1		52 
47	6	1		52 
48	6	1		52 
49	6	1		52 
50	6	1		52 
51	6	1		52 
52	1	0		
************************************************************************
REQUESTS/DURATIONS
jobnr.	mode	dur	R1	R2	N1	N2	N3	N4	
------------------------------------------------------------------------
1	1	0	0	0	0	0	0	0	
2	1	3	4	12	12	18	16	19	
	2	4	4	12	12	18	16	18	
	3	6	4	12	12	17	15	17	
	4	12	4	12	11	16	13	15	
	5	17	4	11	11	15	13	14	
	6	19	4	11	11	14	12	14	
3	1	2	13	6	12	18	18	12	
	2	9	13	5	12	15	16	9	
	3	10	12	5	9	15	16	9	
	4	15	11	5	8	12	15	8	
	5	18	10	5	6	12	14	7	
	6	19	8	5	3	9	14	5	
4	1	1	17	13	6	16	18	17	
	2	6	12	12	6	14	17	16	
	3	7	9	11	5	10	14	15	
	4	9	9	11	5	10	10	12	
	5	12	3	10	4	5	8	12	
	6	20	1	10	4	3	6	9	
5	1	1	13	13	18	20	17	19	
	2	2	12	11	17	13	16	17	
	3	6	12	11	17	10	16	13	
	4	16	11	9	17	9	14	12	
	5	19	9	6	16	4	12	8	
	6	20	7	5	16	1	12	5	
6	1	7	19	15	19	15	14	8	
	2	9	16	14	16	14	13	7	
	3	13	13	12	10	11	13	7	
	4	14	9	11	9	10	13	6	
	5	15	4	11	5	9	11	4	
	6	16	4	9	4	7	11	3	
7	1	4	14	15	20	10	10	12	
	2	6	13	12	19	9	10	12	
	3	12	13	11	17	7	10	10	
	4	14	13	7	16	5	10	10	
	5	19	12	6	16	3	9	8	
	6	20	12	3	14	1	9	7	
8	1	4	14	12	10	10	17	16	
	2	5	13	11	10	9	14	15	
	3	13	12	9	10	8	11	12	
	4	15	12	8	10	7	9	12	
	5	16	12	5	10	5	7	9	
	6	20	11	4	10	5	5	8	
9	1	7	18	18	14	11	15	12	
	2	9	17	16	14	9	15	12	
	3	13	16	14	12	7	14	12	
	4	14	14	12	11	6	14	11	
	5	17	13	10	10	4	13	10	
	6	19	12	9	10	3	12	10	
10	1	5	19	13	18	18	17	16	
	2	9	17	10	17	18	14	14	
	3	10	17	9	16	17	11	11	
	4	11	15	8	16	17	9	8	
	5	12	13	6	15	16	8	8	
	6	13	11	4	15	16	7	6	
11	1	2	15	19	12	16	11	16	
	2	3	15	18	10	15	11	14	
	3	8	15	16	8	13	9	12	
	4	11	15	16	7	13	9	8	
	5	17	15	14	6	11	6	5	
	6	18	15	14	4	9	5	2	
12	1	1	7	14	13	19	17	14	
	2	2	6	14	11	16	16	10	
	3	3	5	12	11	15	15	9	
	4	5	4	12	9	10	14	7	
	5	8	3	10	8	10	12	7	
	6	10	1	10	8	7	11	5	
13	1	1	20	13	16	10	9	10	
	2	5	18	13	13	9	8	10	
	3	12	18	13	11	9	7	10	
	4	13	17	13	8	8	7	10	
	5	14	16	12	6	7	7	10	
	6	19	15	12	6	7	6	10	
14	1	6	17	19	8	2	4	6	
	2	9	15	19	7	2	4	4	
	3	10	14	17	7	2	3	4	
	4	11	13	17	7	2	3	4	
	5	14	10	16	7	2	2	2	
	6	20	7	15	7	2	2	2	
15	1	4	4	18	10	10	8	15	
	2	7	3	16	9	10	7	13	
	3	10	2	16	7	10	5	13	
	4	11	2	15	5	10	3	11	
	5	15	2	14	3	10	2	11	
	6	20	1	14	2	10	1	10	
16	1	4	14	17	10	5	7	12	
	2	9	13	14	8	5	7	11	
	3	12	13	13	7	3	6	10	
	4	14	13	11	6	3	6	10	
	5	17	13	11	5	1	5	8	
	6	18	13	10	2	1	5	8	
17	1	1	17	17	8	10	10	6	
	2	6	15	17	8	10	7	6	
	3	7	11	17	8	10	6	5	
	4	9	9	17	8	10	4	4	
	5	12	6	16	8	10	3	2	
	6	18	5	16	8	10	3	2	
18	1	3	13	11	14	20	19	18	
	2	7	12	11	13	18	17	17	
	3	8	11	11	13	17	16	16	
	4	10	11	11	13	15	16	15	
	5	11	11	11	13	14	14	12	
	6	12	10	11	13	12	14	11	
19	1	3	13	10	15	16	17	18	
	2	9	13	7	15	16	16	17	
	3	11	12	6	15	13	15	16	
	4	15	11	5	15	10	15	14	
	5	17	10	4	15	9	14	13	
	6	19	10	4	15	7	14	13	
20	1	5	9	16	17	14	17	16	
	2	7	9	13	14	14	17	16	
	3	14	9	10	12	14	16	16	
	4	15	9	8	7	14	15	16	
	5	17	9	7	7	14	14	16	
	6	19	9	6	4	14	13	16	
21	1	1	9	6	18	16	17	9	
	2	5	9	6	15	16	15	8	
	3	6	9	5	13	16	12	8	
	4	9	9	5	13	16	11	8	
	5	12	9	4	11	15	8	7	
	6	19	9	4	9	15	7	7	
22	1	5	18	13	17	19	15	15	
	2	9	16	11	15	16	14	14	
	3	10	14	10	15	15	14	13	
	4	11	12	9	14	12	14	12	
	5	15	10	8	13	11	14	11	
	6	16	10	6	12	10	14	10	
23	1	2	18	14	18	15	8	16	
	2	3	15	13	16	13	6	15	
	3	11	13	12	15	13	6	15	
	4	14	11	10	14	11	5	14	
	5	15	7	9	13	10	3	14	
	6	16	7	9	13	10	2	14	
24	1	9	19	11	2	12	10	3	
	2	10	14	11	2	12	9	3	
	3	11	14	10	2	10	8	3	
	4	13	8	10	2	9	7	3	
	5	14	6	9	2	9	7	3	
	6	17	4	9	2	7	6	3	
25	1	3	19	19	13	4	11	16	
	2	4	18	13	12	3	10	16	
	3	11	18	13	12	3	8	16	
	4	12	18	9	12	2	8	16	
	5	15	17	8	11	2	6	16	
	6	16	16	5	11	2	5	16	
26	1	3	18	6	11	15	16	10	
	2	6	18	4	9	15	14	9	
	3	8	18	3	8	15	14	9	
	4	12	18	3	8	15	12	7	
	5	15	17	1	5	14	10	7	
	6	16	17	1	4	14	9	6	
27	1	3	15	17	14	18	16	16	
	2	6	14	17	13	14	16	14	
	3	7	11	17	8	12	16	12	
	4	10	11	17	8	10	15	12	
	5	12	9	17	4	7	14	10	
	6	14	5	17	1	2	14	9	
28	1	4	12	16	18	12	13	20	
	2	7	12	13	17	10	11	15	
	3	11	10	13	17	10	10	12	
	4	16	8	11	16	7	8	8	
	5	19	5	8	16	5	8	4	
	6	20	3	7	15	2	7	3	
29	1	3	10	16	18	17	19	12	
	2	6	9	13	16	16	19	11	
	3	8	8	12	15	16	19	10	
	4	11	8	11	14	15	18	7	
	5	16	6	7	13	14	17	7	
	6	20	5	6	12	13	17	6	
30	1	1	13	11	12	17	15	14	
	2	4	11	11	11	15	14	11	
	3	9	8	11	10	13	14	9	
	4	12	8	11	9	11	11	5	
	5	14	6	11	8	9	11	4	
	6	20	5	11	7	9	8	1	
31	1	2	17	19	19	11	9	16	
	2	3	17	15	17	10	8	12	
	3	9	13	14	17	9	6	12	
	4	10	12	11	15	7	4	9	
	5	13	9	9	13	6	4	9	
	6	16	8	8	12	5	3	6	
32	1	6	9	2	18	13	17	3	
	2	8	8	2	17	9	13	3	
	3	14	8	2	17	7	12	3	
	4	17	8	2	16	5	7	3	
	5	18	8	2	15	5	4	3	
	6	19	8	2	15	1	3	3	
33	1	1	13	12	17	11	6	14	
	2	2	12	11	16	9	5	11	
	3	5	9	10	15	8	5	9	
	4	6	7	9	15	7	5	6	
	5	15	4	9	14	5	5	4	
	6	20	4	8	13	3	5	3	
34	1	2	14	4	9	2	16	18	
	2	5	13	4	9	1	15	17	
	3	6	12	4	7	1	15	17	
	4	9	10	4	6	1	14	16	
	5	10	6	4	5	1	12	16	
	6	11	3	4	2	1	10	16	
35	1	3	17	17	17	16	11	20	
	2	5	13	13	17	15	11	16	
	3	8	12	13	16	13	11	14	
	4	15	7	11	16	13	11	11	
	5	16	5	7	15	12	10	9	
	6	20	3	5	14	10	10	7	
36	1	4	18	15	19	13	4	16	
	2	7	16	12	16	10	4	12	
	3	8	13	11	13	8	4	10	
	4	11	9	8	12	7	4	10	
	5	12	8	6	9	5	4	7	
	6	14	5	3	7	4	4	5	
37	1	2	16	15	13	13	7	7	
	2	5	15	15	10	13	6	5	
	3	6	15	14	7	12	6	5	
	4	8	15	13	6	12	5	4	
	5	14	14	12	3	11	4	3	
	6	16	13	11	3	11	4	3	
38	1	2	17	14	11	15	19	13	
	2	3	14	13	10	14	19	12	
	3	5	14	13	10	13	19	11	
	4	6	13	13	9	11	19	11	
	5	7	11	13	9	9	18	9	
	6	8	10	13	9	9	18	9	
39	1	1	11	6	17	11	19	16	
	2	3	9	6	16	10	16	14	
	3	5	8	6	15	9	15	12	
	4	6	7	6	14	8	13	9	
	5	13	5	6	14	6	11	8	
	6	16	4	6	13	4	10	7	
40	1	7	6	17	13	14	9	17	
	2	10	5	17	12	12	9	15	
	3	11	5	17	11	11	9	14	
	4	12	3	17	9	6	9	12	
	5	13	3	17	6	4	9	10	
	6	19	1	17	5	3	9	9	
41	1	1	17	5	17	6	5	8	
	2	7	15	5	16	5	4	7	
	3	8	14	3	15	4	4	6	
	4	11	14	3	13	4	3	5	
	5	12	13	2	13	3	2	5	
	6	15	12	1	11	2	2	4	
42	1	2	12	19	17	5	19	12	
	2	3	11	18	17	5	17	10	
	3	8	10	18	16	5	14	7	
	4	10	9	16	16	5	9	7	
	5	12	8	15	15	5	7	5	
	6	19	8	15	14	5	4	3	
43	1	2	13	19	19	17	11	17	
	2	5	13	17	19	14	9	15	
	3	9	13	17	19	14	9	11	
	4	10	13	16	18	12	7	10	
	5	13	13	16	18	10	7	7	
	6	17	13	15	17	9	5	4	
44	1	1	14	16	13	16	19	16	
	2	2	12	11	12	16	17	14	
	3	3	8	10	11	16	17	12	
	4	12	5	7	9	16	17	11	
	5	17	5	6	9	15	15	10	
	6	18	2	4	8	15	15	8	
45	1	2	3	20	7	10	15	18	
	2	3	2	16	6	10	14	17	
	3	4	2	13	6	10	14	16	
	4	9	2	9	6	10	14	15	
	5	13	2	5	6	10	14	15	
	6	16	2	4	6	10	14	14	
46	1	1	8	16	13	14	20	19	
	2	2	7	16	13	14	18	17	
	3	5	7	12	13	13	18	14	
	4	6	7	9	12	11	18	11	
	5	13	7	7	11	11	16	9	
	6	16	7	6	11	10	16	8	
47	1	4	19	14	13	12	13	14	
	2	5	18	12	11	11	13	14	
	3	6	18	10	10	9	11	13	
	4	11	17	9	10	8	9	12	
	5	15	17	7	7	7	8	12	
	6	19	16	7	7	6	7	11	
48	1	1	14	11	10	6	2	13	
	2	2	10	10	9	4	2	13	
	3	5	10	9	9	3	2	13	
	4	9	8	8	8	2	1	12	
	5	19	5	8	8	1	1	12	
	6	20	3	7	8	1	1	11	
49	1	1	13	2	17	10	14	5	
	2	10	12	2	15	8	14	4	
	3	12	11	2	15	6	12	4	
	4	15	11	1	15	6	11	3	
	5	17	11	1	14	4	10	3	
	6	18	10	1	13	2	10	3	
50	1	7	16	17	11	7	17	14	
	2	10	16	17	9	7	17	12	
	3	14	16	16	6	7	15	11	
	4	16	16	15	6	7	13	8	
	5	17	16	14	4	6	11	7	
	6	20	16	14	2	6	9	5	
51	1	1	16	17	16	19	8	10	
	2	8	14	15	16	18	7	9	
	3	9	13	14	15	18	7	8	
	4	12	12	13	15	18	7	6	
	5	13	10	12	13	18	7	5	
	6	14	9	12	13	18	7	5	
52	1	0	0	0	0	0	0	0	
************************************************************************

 RESOURCE AVAILABILITIES 
	R 1	R 2	N 1	N 2	N 3	N 4
	47	46	572	500	539	522

************************************************************************
