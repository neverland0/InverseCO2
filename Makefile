cxx = g++
LIBS := -lnlopt -lm
optPara : optimizeHyperpara.cc
	$(cxx) -o optPara optimizeHyperpara.cc $(LIBS)
.PHONY : clean
clean :
	rm optPara
