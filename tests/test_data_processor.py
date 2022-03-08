from src.processors.base_processor import ChnsenticorpDataProcessor, XnliCNDatProcessor



def test_chnsenticorp():
    processor = ChnsenticorpDataProcessor()
    dataset = processor.get_dev_dataset()
    
    assert dataset is not None