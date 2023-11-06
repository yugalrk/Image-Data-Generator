from exercise0_material.src_to_implement.generator import ImageGenerator

if __name__ == "__main__":
    DataGen = ImageGenerator(
        r'C:\Users\YUGAL\Desktop\exercise0_material\exercise0_material\src_to_implement\exercise_data',
        r'C:\Users\YUGAL\Desktop\exercise0_material\exercise0_material\src_to_implement\Labels.json',
        10,
        (150, 150, 3))
    DataGen.next()
    DataGen.current_epoch()
    DataGen.show()
