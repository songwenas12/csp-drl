jobs  (incl. supersource/sink ):	52
RESOURCES
- renewable                 : 4 R
- nonrenewable              : 2 N
- doubly constrained        : 0 D
************************************************************************
PRECEDENCE RELATIONS:
jobnr.    #modes  #successors   successors
1	1	10		2 3 4 5 6 7 8 9 10 13 
2	9	9		29 22 20 19 17 16 14 12 11 
3	9	9		37 22 21 19 17 16 14 12 11 
4	9	8		35 22 20 19 16 15 14 11 
5	9	8		35 29 22 21 19 16 14 11 
6	9	8		37 35 28 23 22 19 15 14 
7	9	7		37 35 28 23 19 15 14 
8	9	6		35 30 29 22 20 19 
9	9	11		51 50 49 37 35 33 30 28 27 25 24 
10	9	8		51 37 32 28 27 26 23 18 
11	9	7		51 32 28 27 25 23 18 
12	9	12		51 50 49 47 46 40 35 32 30 28 27 26 
13	9	2		32 14 
14	9	9		51 50 49 40 33 31 30 27 25 
15	9	7		51 50 49 31 29 25 24 
16	9	9		51 50 49 48 45 33 30 27 24 
17	9	9		48 45 39 35 33 31 30 27 24 
18	9	10		50 49 48 45 39 36 34 33 31 30 
19	9	6		51 40 36 33 31 26 
20	9	8		47 46 40 39 37 34 32 31 
21	9	8		50 49 48 46 40 39 30 27 
22	9	7		50 49 48 47 45 39 27 
23	9	8		49 47 46 45 40 39 36 34 
24	9	4		46 40 34 32 
25	9	6		48 47 45 39 38 36 
26	9	6		48 45 44 43 39 34 
27	9	3		44 36 34 
28	9	3		44 36 31 
29	9	5		47 46 45 43 38 
30	9	4		44 43 42 38 
31	9	3		43 42 38 
32	9	2		44 36 
33	9	3		47 46 42 
34	9	2		42 38 
35	9	2		42 38 
36	9	2		43 42 
37	9	2		45 41 
38	9	1		41 
39	9	1		41 
40	9	1		41 
41	9	1		52 
42	9	1		52 
43	9	1		52 
44	9	1		52 
45	9	1		52 
46	9	1		52 
47	9	1		52 
48	9	1		52 
49	9	1		52 
50	9	1		52 
51	9	1		52 
52	1	0		
************************************************************************
REQUESTS/DURATIONS
jobnr.	mode	dur	R1	R2	R3	R4	N1	N2	
------------------------------------------------------------------------
1	1	0	0	0	0	0	0	0	
2	1	6	5	3	4	2	25	16	
	2	10	4	3	4	2	25	14	
	3	11	4	3	3	2	23	12	
	4	16	4	2	3	2	22	10	
	5	18	3	2	3	1	22	10	
	6	22	3	2	2	1	21	8	
	7	24	3	2	2	1	20	7	
	8	28	2	1	1	1	18	5	
	9	30	2	1	1	1	18	3	
3	1	2	1	4	3	2	28	16	
	2	4	1	4	3	2	26	15	
	3	10	1	4	3	2	26	14	
	4	11	1	4	3	2	24	14	
	5	12	1	4	3	2	24	13	
	6	21	1	4	3	1	23	13	
	7	23	1	4	3	1	22	13	
	8	26	1	4	3	1	20	12	
	9	27	1	4	3	1	20	11	
4	1	2	5	2	1	5	12	28	
	2	3	4	2	1	4	12	27	
	3	4	4	2	1	4	11	27	
	4	5	4	2	1	4	8	26	
	5	12	3	2	1	4	8	23	
	6	13	3	2	1	4	5	23	
	7	20	2	2	1	4	5	22	
	8	22	2	2	1	4	2	21	
	9	23	2	2	1	4	2	20	
5	1	2	1	5	1	4	16	10	
	2	6	1	4	1	4	13	11	
	3	7	1	4	1	4	13	10	
	4	12	1	4	1	3	10	10	
	5	17	1	4	1	2	8	10	
	6	19	1	4	1	2	7	9	
	7	20	1	4	1	2	5	9	
	8	22	1	4	1	1	5	9	
	9	28	1	4	1	1	3	9	
6	1	2	4	4	4	5	26	21	
	2	3	4	3	4	4	23	19	
	3	7	4	3	4	4	23	18	
	4	11	4	2	4	4	20	16	
	5	19	4	2	4	4	20	15	
	6	20	4	2	4	4	19	15	
	7	23	4	2	4	4	17	12	
	8	25	4	1	4	4	16	10	
	9	29	4	1	4	4	14	9	
7	1	2	3	3	4	4	19	27	
	2	4	3	2	4	3	17	25	
	3	14	3	2	4	3	16	25	
	4	15	3	2	3	3	13	23	
	5	19	3	1	3	3	10	22	
	6	23	3	1	3	3	10	20	
	7	27	3	1	2	3	5	19	
	8	28	3	1	2	3	4	17	
	9	29	3	1	2	3	2	16	
8	1	2	5	3	4	3	28	27	
	2	4	4	2	3	3	28	24	
	3	5	4	2	3	3	26	24	
	4	15	3	2	3	3	22	20	
	5	16	3	2	3	3	21	16	
	6	24	3	2	3	3	19	16	
	7	26	3	2	3	3	18	12	
	8	27	2	2	3	3	14	11	
	9	29	2	2	3	3	14	9	
9	1	4	3	3	5	5	28	26	
	2	9	3	3	4	5	24	25	
	3	10	3	3	4	5	21	24	
	4	11	3	3	4	5	18	24	
	5	12	3	3	4	5	16	22	
	6	17	3	3	3	5	13	21	
	7	18	3	3	3	5	9	20	
	8	28	3	3	3	5	7	19	
	9	29	3	3	3	5	5	19	
10	1	5	5	4	3	4	2	27	
	2	6	4	4	3	4	2	26	
	3	7	4	4	3	4	2	23	
	4	19	3	3	3	4	2	23	
	5	20	3	3	3	4	2	21	
	6	25	2	3	3	3	2	19	
	7	28	2	3	3	3	2	18	
	8	29	1	2	3	3	2	16	
	9	30	1	2	3	3	2	15	
11	1	2	2	5	4	4	26	25	
	2	6	1	4	4	3	25	22	
	3	9	1	4	4	3	24	18	
	4	16	1	3	4	3	23	16	
	5	22	1	3	4	2	23	12	
	6	23	1	3	4	2	22	9	
	7	28	1	3	4	2	22	7	
	8	29	1	2	4	2	22	6	
	9	30	1	2	4	2	21	3	
12	1	2	2	2	2	4	23	4	
	2	3	1	2	2	4	22	3	
	3	4	1	2	2	3	20	3	
	4	5	1	2	2	3	20	2	
	5	12	1	2	2	3	19	2	
	6	13	1	1	1	2	17	2	
	7	16	1	1	1	2	16	2	
	8	17	1	1	1	1	15	1	
	9	18	1	1	1	1	14	1	
13	1	6	4	5	1	2	27	9	
	2	7	3	4	1	1	24	8	
	3	13	3	4	1	1	19	6	
	4	15	2	4	1	1	18	5	
	5	23	2	4	1	1	15	5	
	6	24	2	4	1	1	11	3	
	7	25	1	4	1	1	10	3	
	8	27	1	4	1	1	7	1	
	9	30	1	4	1	1	4	1	
14	1	6	5	5	5	5	13	6	
	2	7	4	4	4	4	10	5	
	3	16	3	4	4	4	9	4	
	4	18	3	4	3	4	9	4	
	5	19	2	3	3	4	6	3	
	6	20	2	3	3	4	6	2	
	7	22	1	3	2	4	3	2	
	8	25	1	3	2	4	3	1	
	9	26	1	3	2	4	2	2	
15	1	1	4	2	3	3	23	29	
	2	7	4	2	2	3	22	29	
	3	8	4	2	2	3	21	29	
	4	9	4	2	2	3	19	29	
	5	10	4	2	1	3	19	29	
	6	12	4	2	1	2	18	28	
	7	14	4	2	1	2	17	28	
	8	25	4	2	1	2	15	28	
	9	30	4	2	1	2	15	27	
16	1	2	1	5	2	4	29	27	
	2	5	1	4	1	4	28	25	
	3	11	1	4	1	4	27	24	
	4	13	1	3	1	4	27	22	
	5	14	1	2	1	4	25	21	
	6	15	1	2	1	4	25	20	
	7	17	1	1	1	4	25	19	
	8	19	1	1	1	4	23	17	
	9	24	1	1	1	4	23	15	
17	1	5	3	2	3	4	19	24	
	2	6	3	2	3	3	17	21	
	3	8	3	2	3	3	16	18	
	4	9	3	2	3	3	15	17	
	5	12	3	2	2	3	13	13	
	6	22	3	2	2	3	12	10	
	7	25	3	2	2	3	11	8	
	8	26	3	2	2	3	11	5	
	9	28	3	2	2	3	9	3	
18	1	11	1	3	5	5	18	27	
	2	14	1	3	4	5	18	27	
	3	15	1	3	4	5	14	27	
	4	20	1	3	3	5	14	26	
	5	22	1	3	3	5	11	26	
	6	23	1	3	3	5	11	25	
	7	24	1	3	3	5	8	26	
	8	26	1	3	2	5	5	25	
	9	27	1	3	2	5	5	24	
19	1	2	5	3	2	5	10	27	
	2	3	4	3	1	4	10	24	
	3	9	4	3	1	4	10	20	
	4	11	4	3	1	4	10	19	
	5	15	4	2	1	4	9	16	
	6	23	4	2	1	4	9	14	
	7	24	4	2	1	4	9	13	
	8	26	4	1	1	4	8	10	
	9	30	4	1	1	4	8	6	
20	1	5	2	4	4	2	9	22	
	2	9	1	4	3	2	9	22	
	3	13	1	4	3	2	7	22	
	4	14	1	3	3	2	7	22	
	5	15	1	3	3	2	5	21	
	6	23	1	3	3	2	4	21	
	7	25	1	2	3	2	4	21	
	8	27	1	2	3	2	3	21	
	9	29	1	2	3	2	2	21	
21	1	1	1	4	3	5	26	24	
	2	2	1	3	2	4	25	21	
	3	5	1	3	2	4	25	20	
	4	6	1	3	2	3	24	18	
	5	7	1	2	2	3	24	15	
	6	15	1	2	1	3	24	13	
	7	19	1	2	1	2	23	13	
	8	22	1	2	1	2	23	10	
	9	25	1	2	1	2	22	7	
22	1	8	3	4	5	1	27	30	
	2	9	3	4	5	1	25	26	
	3	10	3	3	5	1	24	24	
	4	13	3	3	5	1	22	20	
	5	20	3	2	5	1	17	15	
	6	21	2	2	5	1	17	13	
	7	22	2	1	5	1	14	9	
	8	28	2	1	5	1	10	8	
	9	29	2	1	5	1	10	5	
23	1	2	3	4	1	4	17	23	
	2	8	3	4	1	3	17	23	
	3	10	3	4	1	3	17	22	
	4	14	3	4	1	3	17	21	
	5	18	3	4	1	2	16	22	
	6	19	3	4	1	2	16	21	
	7	20	3	4	1	2	16	20	
	8	27	3	4	1	2	15	20	
	9	28	3	4	1	2	15	19	
24	1	2	5	4	4	4	17	27	
	2	3	4	3	3	4	16	26	
	3	4	4	3	3	4	16	25	
	4	5	4	3	3	4	14	26	
	5	17	4	3	3	3	14	26	
	6	22	4	3	3	3	12	25	
	7	23	4	3	3	2	12	25	
	8	26	4	3	3	2	10	25	
	9	29	4	3	3	2	10	24	
25	1	4	3	2	4	3	21	19	
	2	6	2	1	3	3	19	16	
	3	11	2	1	3	3	16	15	
	4	13	2	1	3	3	14	12	
	5	19	1	1	3	3	14	10	
	6	20	1	1	3	3	12	10	
	7	22	1	1	3	3	9	9	
	8	23	1	1	3	3	8	6	
	9	27	1	1	3	3	6	4	
26	1	3	3	4	4	5	17	13	
	2	7	3	4	4	4	17	12	
	3	8	3	4	4	4	16	10	
	4	19	3	4	4	4	15	10	
	5	20	2	4	4	4	15	7	
	6	21	2	4	4	3	14	7	
	7	23	2	4	4	3	14	4	
	8	29	2	4	4	3	14	3	
	9	30	2	4	4	3	13	2	
27	1	2	5	4	4	1	13	30	
	2	5	4	3	4	1	11	27	
	3	8	3	3	4	1	11	23	
	4	17	3	3	4	1	9	18	
	5	18	2	2	3	1	8	17	
	6	21	2	2	3	1	7	14	
	7	22	1	2	3	1	6	13	
	8	25	1	1	2	1	6	7	
	9	29	1	1	2	1	5	5	
28	1	3	4	5	4	2	23	15	
	2	10	4	4	3	2	23	15	
	3	12	4	4	3	2	23	13	
	4	18	4	4	3	2	22	12	
	5	19	3	4	3	2	22	12	
	6	20	3	3	3	2	22	11	
	7	21	3	3	3	2	22	10	
	8	26	2	3	3	2	21	10	
	9	28	2	3	3	2	21	9	
29	1	6	5	3	4	2	24	30	
	2	9	4	3	4	2	20	25	
	3	13	4	3	4	2	19	24	
	4	14	3	3	4	2	16	21	
	5	16	2	3	4	2	16	20	
	6	17	2	3	3	2	13	17	
	7	19	1	3	3	2	10	13	
	8	21	1	3	3	2	10	12	
	9	30	1	3	3	2	6	11	
30	1	5	5	3	1	3	11	11	
	2	8	4	3	1	3	10	9	
	3	14	4	3	1	3	8	9	
	4	16	4	3	1	3	8	8	
	5	19	3	3	1	3	6	7	
	6	20	3	3	1	3	5	7	
	7	26	3	3	1	3	4	7	
	8	29	3	3	1	3	4	5	
	9	30	3	3	1	3	3	5	
31	1	1	4	4	4	5	27	9	
	2	16	3	4	4	4	27	8	
	3	18	3	3	4	4	25	6	
	4	19	3	3	3	4	24	5	
	5	20	2	3	2	3	24	4	
	6	25	2	2	2	3	23	3	
	7	26	2	1	1	3	22	2	
	8	29	2	1	1	3	20	2	
	9	30	2	1	1	3	20	1	
32	1	3	3	3	4	3	30	23	
	2	4	2	3	3	2	28	22	
	3	7	2	3	3	2	27	21	
	4	10	2	3	3	2	26	20	
	5	14	2	3	3	2	23	18	
	6	15	1	3	3	2	23	17	
	7	18	1	3	3	2	21	16	
	8	19	1	3	3	2	21	14	
	9	28	1	3	3	2	19	13	
33	1	2	2	4	4	5	18	23	
	2	3	2	3	3	4	17	23	
	3	8	2	3	3	4	17	22	
	4	9	2	3	3	4	16	22	
	5	10	2	3	3	3	13	22	
	6	14	2	3	3	3	12	22	
	7	17	2	3	3	3	10	22	
	8	20	2	3	3	2	9	21	
	9	21	2	3	3	2	8	21	
34	1	1	5	5	3	4	26	22	
	2	9	4	4	3	3	24	19	
	3	11	4	4	3	3	23	17	
	4	13	4	4	3	3	23	15	
	5	14	3	3	2	3	22	15	
	6	15	3	3	2	3	22	11	
	7	16	3	3	2	3	20	9	
	8	18	3	3	1	3	20	7	
	9	27	3	3	1	3	19	7	
35	1	3	3	3	4	4	22	21	
	2	4	2	2	4	4	20	21	
	3	12	2	2	4	4	19	19	
	4	13	2	2	4	4	18	19	
	5	18	2	1	4	4	18	18	
	6	22	2	1	4	4	16	16	
	7	25	2	1	4	4	16	15	
	8	28	2	1	4	4	15	15	
	9	29	2	1	4	4	14	14	
36	1	3	4	4	4	2	28	10	
	2	5	4	3	4	2	25	9	
	3	8	4	3	4	2	24	9	
	4	11	4	3	4	2	23	9	
	5	12	4	2	3	2	23	8	
	6	17	3	2	3	2	20	7	
	7	27	3	2	3	2	20	6	
	8	28	3	1	3	2	19	6	
	9	29	3	1	3	2	18	6	
37	1	7	2	4	3	3	29	22	
	2	11	2	4	2	2	26	18	
	3	12	2	4	2	2	25	18	
	4	17	2	4	2	2	22	15	
	5	18	2	4	2	2	20	14	
	6	19	2	3	1	1	20	12	
	7	20	2	3	1	1	17	11	
	8	21	2	3	1	1	14	9	
	9	28	2	3	1	1	14	6	
38	1	7	4	4	2	3	13	16	
	2	13	3	4	2	2	13	13	
	3	18	3	4	2	2	11	13	
	4	19	3	3	2	2	10	11	
	5	21	2	2	1	2	8	10	
	6	22	2	2	1	1	8	9	
	7	23	2	2	1	1	6	8	
	8	24	1	1	1	1	4	6	
	9	27	1	1	1	1	4	5	
39	1	3	5	2	3	5	24	14	
	2	10	4	2	3	4	22	14	
	3	13	3	2	3	4	21	14	
	4	14	3	2	3	4	19	14	
	5	18	3	2	2	3	16	14	
	6	22	2	2	2	3	16	14	
	7	23	1	2	2	3	15	14	
	8	24	1	2	1	3	13	14	
	9	28	1	2	1	3	11	14	
40	1	8	4	4	1	3	30	24	
	2	9	4	4	1	3	29	24	
	3	14	4	3	1	3	29	23	
	4	18	3	3	1	3	29	22	
	5	19	3	2	1	3	28	22	
	6	20	3	2	1	3	28	21	
	7	22	2	2	1	3	28	20	
	8	28	2	1	1	3	27	20	
	9	29	2	1	1	3	27	19	
41	1	8	5	3	3	5	18	16	
	2	13	4	2	3	4	16	14	
	3	14	4	2	3	4	16	12	
	4	16	4	2	3	3	16	12	
	5	19	3	2	2	3	15	10	
	6	20	3	2	2	3	14	8	
	7	23	2	2	1	3	14	8	
	8	25	2	2	1	2	14	6	
	9	26	2	2	1	2	13	5	
42	1	2	5	2	3	4	28	28	
	2	11	4	1	3	4	26	27	
	3	14	4	1	3	4	24	27	
	4	17	4	1	3	3	24	25	
	5	21	4	1	3	3	22	25	
	6	23	4	1	3	3	21	24	
	7	26	4	1	3	2	18	23	
	8	27	4	1	3	2	18	22	
	9	29	4	1	3	2	15	22	
43	1	2	4	4	5	5	27	16	
	2	8	4	4	4	4	23	15	
	3	12	4	3	4	4	21	15	
	4	19	4	3	4	3	17	14	
	5	24	4	3	4	3	17	13	
	6	25	4	2	4	2	13	13	
	7	28	4	1	4	1	10	12	
	8	29	4	1	4	1	8	12	
	9	30	4	1	4	1	4	12	
44	1	6	4	1	5	4	19	10	
	2	11	3	1	4	4	17	9	
	3	17	3	1	4	4	17	7	
	4	18	3	1	3	4	16	6	
	5	20	3	1	3	4	15	5	
	6	21	3	1	3	4	13	5	
	7	27	3	1	2	4	13	3	
	8	28	3	1	2	4	12	2	
	9	29	3	1	2	4	11	2	
45	1	8	4	3	1	2	29	17	
	2	9	4	3	1	2	28	15	
	3	13	4	3	1	2	28	14	
	4	16	4	3	1	2	28	13	
	5	17	4	3	1	2	28	11	
	6	21	3	3	1	2	28	9	
	7	25	3	3	1	2	28	8	
	8	26	3	3	1	2	28	7	
	9	29	3	3	1	2	28	6	
46	1	3	4	3	3	4	17	21	
	2	5	4	2	3	4	16	20	
	3	10	4	2	3	4	14	20	
	4	12	4	2	3	4	11	20	
	5	13	4	2	3	3	11	19	
	6	25	4	2	3	3	8	19	
	7	26	4	2	3	3	5	19	
	8	27	4	2	3	3	3	19	
	9	29	4	2	3	3	2	19	
47	1	1	5	4	1	1	20	27	
	2	4	5	3	1	1	19	23	
	3	5	5	3	1	1	19	19	
	4	8	5	3	1	1	18	19	
	5	14	5	3	1	1	16	15	
	6	16	5	3	1	1	15	11	
	7	17	5	3	1	1	15	8	
	8	25	5	3	1	1	13	7	
	9	27	5	3	1	1	13	1	
48	1	9	4	4	1	3	18	23	
	2	11	4	3	1	3	16	22	
	3	13	4	3	1	3	16	21	
	4	14	4	3	1	2	16	22	
	5	16	3	2	1	2	15	22	
	6	17	3	2	1	2	14	21	
	7	20	3	1	1	2	14	21	
	8	21	3	1	1	1	13	21	
	9	22	3	1	1	1	13	20	
49	1	7	3	2	5	3	28	18	
	2	8	2	1	5	2	27	16	
	3	18	2	1	5	2	26	14	
	4	19	2	1	5	2	26	13	
	5	25	2	1	5	2	24	11	
	6	26	1	1	5	1	24	9	
	7	27	1	1	5	1	23	7	
	8	29	1	1	5	1	23	4	
	9	30	1	1	5	1	22	3	
50	1	2	4	4	3	3	14	21	
	2	5	3	3	3	3	13	21	
	3	6	3	3	3	3	12	21	
	4	14	2	2	3	3	12	20	
	5	18	2	2	3	3	11	20	
	6	23	2	2	2	3	10	20	
	7	24	2	1	2	3	10	19	
	8	27	1	1	2	3	9	19	
	9	29	1	1	2	3	9	18	
51	1	1	5	4	2	4	22	14	
	2	6	4	4	2	3	20	14	
	3	9	4	4	2	3	20	13	
	4	16	4	3	2	3	19	12	
	5	21	3	3	2	2	18	12	
	6	22	3	2	2	2	17	11	
	7	27	3	2	2	2	15	10	
	8	28	3	1	2	1	15	10	
	9	29	3	1	2	1	14	9	
52	1	0	0	0	0	0	0	0	
************************************************************************

 RESOURCE AVAILABILITIES 
	R 1	R 2	R 3	R 4	N 1	N 2
	34	30	29	29	710	655

************************************************************************
