#ifndef ELEMENT_H
#define ELEMENT_H

class Element {
	int m;
	int value;
public:
	Element(int m, int value);
	bool operator()(const Element* lhs, const Element* rhs) const  { 
		return lhs->m < rhs->m;
	}
};

#endif
