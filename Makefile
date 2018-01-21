EXTERNAL_DIR=$(CURDIR)/external
MCF_SOURCE_DIR=$(EXTERNAL_DIR)/mcf
MCF_BUILD_DIR=$(CURDIR)/build/mcf
PYMOTUTILS_SOURCE_DIR=$(EXTERNAL_DIR)/pymotutils

all: pymotutils generate_detections.py mcf.so

pymotutils:
	ln -s $(PYMOTUTILS_SOURCE_DIR)/pymotutils $(CURDIR)

generate_detections.py:
	wget https://raw.githubusercontent.com/nwojke/deep_sort/master/generate_detections.py

mcf.so:
	mkdir -p $(MCF_BUILD_DIR)
	cd $(MCF_BUILD_DIR) && \
		make -f $(MCF_SOURCE_DIR)/Makefile-external pybind11
	cd $(MCF_BUILD_DIR) && \
		cmake $(MCF_SOURCE_DIR) \
		-DCMAKE_BUILD_TYPE=RELEASE \
		-DMCF_BUILD_PYTHON=ON \
		-DMCF_USE_Lemon=OFF \
		-DMCF_USE_Clp=OFF \
		-DMCF_BUILD_EXAMPLES=OFF \
		-DMCF_BUILD_STATIC=OFF \
		-DMCF_BUILD_SHARED=OFF
	cd $(MCF_BUILD_DIR) && \
		make
	cp $(MCF_BUILD_DIR)/python_lib/mcf*.so $(CURDIR)/mcf.so

clean:
	rm -rf $(CURDIR)/build
	rm $(CURDIR)/mcf.so
	rm $(CURDIR)/pymotutils
	rm $(CURDIR)/generate_detections.py
