#include "element.h"

Element::Element(int m, double value) {
	this->m = m;
	this->value = value;
}

double Element::getValue() {
	return this->value;
}

int Element::getRow() {
	return this->m;
}
