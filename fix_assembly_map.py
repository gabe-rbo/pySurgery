with open("pysurgery/algebraic_surgery.py", "r") as f:
    content = f.read()

new_content = content.replace("return WallGroupL(dimension=self.domain.dimension, pi=pi_1_group).compute_obstruction(self.domain.chain_complex)", 
                              "return WallGroupL(dimension=self.domain.dimension, pi=pi_1_group).compute_obstruction(None)")

with open("pysurgery/algebraic_surgery.py", "w") as f:
    f.write(new_content)
