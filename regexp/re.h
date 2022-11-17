/* Each State represents one of the following three NFA fragments, depending on the value of c:
c < 256: single choice 
c == 256: split choice
c == 257: match
*/
struct State {
	int c;
	State *out;
	State *out1;
	int lastlist;
};

// struct Frag
// {
// 	State *start;
// 	 *out;
// };

