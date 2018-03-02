#include "element.h"

Element::Element(int m, int value) {
	this->m = m;
	this->value = value;
}

double Element::getValue() {
	return this->value;
}

int Element::getRow() {
	return this->m;
}
