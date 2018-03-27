#ifndef ELEMENT_H
#define ELEMENT_H

class Element {
	int m;
	double value;
public:
	Element(int m, double value);
	double getValue();
	int getRow();
	bool operator()(const Element* lhs, const Element* rhs) const  { 
		return lhs->m < rhs->m;
	}
};

#endif
