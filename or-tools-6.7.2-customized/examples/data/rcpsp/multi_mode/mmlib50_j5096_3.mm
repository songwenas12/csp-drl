jobs  (incl. supersource/sink ):	52
RESOURCES
- renewable                 : 2 R
- nonrenewable              : 2 N
- doubly constrained        : 0 D
************************************************************************
PRECEDENCE RELATIONS:
jobnr.    #modes  #successors   successors
1	1	7		2 3 4 5 6 11 13 
2	3	5		19 12 10 9 7 
3	3	3		19 17 8 
4	3	2		12 7 
5	3	2		18 8 
6	3	2		10 9 
7	3	7		23 22 20 18 17 16 14 
8	3	2		14 10 
9	3	5		23 22 20 17 14 
10	3	6		23 22 21 20 16 15 
11	3	4		22 20 18 14 
12	3	3		22 17 16 
13	3	3		22 17 16 
14	3	4		28 24 21 15 
15	3	4		37 27 26 25 
16	3	4		37 28 27 26 
17	3	2		24 21 
18	3	4		37 33 27 24 
19	3	4		37 29 28 25 
20	3	3		29 28 24 
21	3	3		37 29 25 
22	3	3		37 30 27 
23	3	3		33 30 27 
24	3	1		25 
25	3	3		36 34 30 
26	3	2		36 29 
27	3	4		36 35 34 31 
28	3	3		39 36 30 
29	3	5		42 39 35 33 31 
30	3	3		42 35 31 
31	3	2		44 32 
32	3	5		46 45 43 41 38 
33	3	5		46 44 43 41 38 
34	3	3		42 39 38 
35	3	3		46 43 38 
36	3	2		42 40 
37	3	2		43 40 
38	3	2		49 40 
39	3	3		48 46 44 
40	3	4		51 50 48 47 
41	3	4		51 50 49 48 
42	3	2		48 46 
43	3	2		49 47 
44	3	2		49 47 
45	3	2		50 47 
46	3	1		47 
47	3	1		52 
48	3	1		52 
49	3	1		52 
50	3	1		52 
51	3	1		52 
52	1	0		
************************************************************************
REQUESTS/DURATIONS
jobnr.	mode	dur	R1	R2	N1	N2	
------------------------------------------------------------------------
1	1	0	0	0	0	0	
2	1	5	4	5	3	2	
	2	9	4	4	2	1	
	3	10	2	4	1	1	
3	1	1	5	4	6	7	
	2	6	3	3	4	7	
	3	8	3	3	3	7	
4	1	3	9	8	8	9	
	2	6	6	7	7	9	
	3	7	4	7	4	9	
5	1	1	8	5	7	6	
	2	4	5	3	6	6	
	3	7	3	2	2	1	
6	1	3	7	4	9	7	
	2	8	5	4	6	6	
	3	9	4	4	3	5	
7	1	3	8	6	4	4	
	2	5	5	5	3	2	
	3	7	4	3	3	2	
8	1	3	4	6	3	3	
	2	8	4	3	2	2	
	3	10	3	2	1	2	
9	1	8	5	6	6	5	
	2	9	4	3	4	3	
	3	10	4	3	4	2	
10	1	4	2	6	8	3	
	2	5	1	5	4	2	
	3	8	1	4	3	2	
11	1	1	4	3	7	6	
	2	7	3	3	6	5	
	3	10	3	3	5	5	
12	1	3	9	9	5	9	
	2	5	7	9	4	9	
	3	7	4	9	4	9	
13	1	2	2	5	7	4	
	2	5	1	4	5	4	
	3	10	1	3	5	4	
14	1	1	8	5	3	4	
	2	2	7	5	3	4	
	3	7	7	4	3	1	
15	1	4	5	7	8	9	
	2	8	3	7	7	9	
	3	10	3	1	7	8	
16	1	2	6	9	4	7	
	2	3	4	5	2	4	
	3	8	3	2	2	2	
17	1	3	8	8	9	5	
	2	5	7	5	8	5	
	3	6	6	3	7	4	
18	1	4	7	3	5	6	
	2	8	6	3	5	5	
	3	9	5	3	3	5	
19	1	1	4	8	9	9	
	2	5	4	4	7	4	
	3	6	4	3	7	2	
20	1	1	8	8	3	5	
	2	3	6	7	3	5	
	3	10	6	6	2	5	
21	1	5	6	8	7	8	
	2	9	5	5	4	7	
	3	10	5	2	3	7	
22	1	1	8	7	8	8	
	2	8	4	5	6	7	
	3	9	2	3	5	6	
23	1	1	7	8	9	8	
	2	4	7	4	9	7	
	3	5	7	2	9	7	
24	1	2	6	8	5	5	
	2	9	4	7	4	5	
	3	10	3	7	3	5	
25	1	2	4	6	6	5	
	2	4	3	4	4	4	
	3	5	3	2	3	4	
26	1	1	1	6	6	8	
	2	4	1	5	6	7	
	3	8	1	2	3	6	
27	1	7	7	5	7	7	
	2	8	6	3	5	5	
	3	9	6	3	3	5	
28	1	5	8	8	7	5	
	2	7	7	7	6	3	
	3	8	5	5	2	1	
29	1	1	8	7	7	8	
	2	6	6	5	4	3	
	3	10	4	4	3	1	
30	1	2	6	3	6	3	
	2	7	6	2	5	3	
	3	9	5	2	4	2	
31	1	3	1	5	2	1	
	2	9	1	3	2	1	
	3	10	1	1	2	1	
32	1	2	5	3	9	6	
	2	6	4	3	8	3	
	3	7	4	2	8	1	
33	1	1	9	7	10	7	
	2	2	9	7	8	5	
	3	9	9	7	7	4	
34	1	1	7	8	7	6	
	2	5	7	6	6	5	
	3	9	5	5	6	4	
35	1	2	6	6	8	4	
	2	3	5	5	4	3	
	3	4	4	4	4	2	
36	1	4	7	5	2	8	
	2	5	4	4	2	6	
	3	6	1	4	2	4	
37	1	1	9	8	9	7	
	2	5	7	8	9	7	
	3	6	7	8	9	6	
38	1	1	5	6	10	6	
	2	5	4	3	8	4	
	3	10	4	3	7	4	
39	1	1	1	5	2	9	
	2	2	1	4	2	8	
	3	6	1	3	2	8	
40	1	4	7	4	4	3	
	2	5	5	3	3	2	
	3	6	4	3	3	1	
41	1	6	9	10	7	2	
	2	9	4	8	7	2	
	3	10	2	5	5	2	
42	1	2	7	5	4	3	
	2	8	5	5	3	3	
	3	9	4	4	3	2	
43	1	5	10	9	6	4	
	2	7	9	7	6	2	
	3	8	9	7	6	1	
44	1	3	10	6	4	8	
	2	5	8	3	4	7	
	3	7	5	2	3	7	
45	1	4	7	10	8	6	
	2	6	4	8	8	6	
	3	10	3	6	8	3	
46	1	6	9	7	7	6	
	2	8	8	5	7	5	
	3	10	7	4	5	2	
47	1	2	7	6	10	8	
	2	3	3	5	5	7	
	3	10	3	3	5	5	
48	1	4	10	5	7	7	
	2	5	8	5	4	6	
	3	6	5	2	4	6	
49	1	8	7	3	5	5	
	2	9	5	2	4	2	
	3	10	1	2	4	1	
50	1	4	6	5	7	4	
	2	5	4	5	6	3	
	3	9	1	5	6	2	
51	1	2	3	8	5	8	
	2	6	2	8	5	7	
	3	9	2	8	3	7	
52	1	0	0	0	0	0	
************************************************************************

 RESOURCE AVAILABILITIES 
	R 1	R 2	N 1	N 2
	29	28	288	268

************************************************************************
