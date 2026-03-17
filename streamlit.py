# import streamlit as st

# def main():
#     st.title("My First Streamlit App")
#     st.write("Hi Students")

# if __name__ == "__main__":
#     main()







# import streamlit as st

# def main():
#     st.title("Streamlit – Step 2")
#     st.write("Hi Students")

#     name = st.text_input("Enter your name:")

#     st.write("You typed:", name)

# if __name__ == "__main__":
#     main()














# import streamlit as st

# def main():
#     st.title("Streamlit – Step 3")
#     st.write("Hi Students")

#     name = st.text_input("Enter your name:")

#     if st.button("Greet "):
#         st.write("Hello", name)

# if __name__ == "__main__":
#     main()















# import streamlit as st

# def main():
#     st.title("Streamlit – Step 4")
#     st.write("Hi Students")

#     name = st.text_input("Enter your name:")

#     st.write("Enter two numbers to add:")

#     a = st.number_input("First number:", value=0)
#     b = st.number_input("Second number:", value=0)

#     if st.button("Calculate Sum"):
#         total = a + b
#         st.write("Hello", name)
#         st.write("Sum of the two numbers is:", total)

# if __name__ == "__main__":
#     main()















import streamlit as st

def main():
    st.title("Streamlit – Step 5")
    st.write("Hi Students")

    name = st.text_input("Enter your name:")

    a = st.number_input("First number:", value=0)
    b = st.number_input("Second number:", value=0)

    operation = st.selectbox(
        "Choose operation:",
        ["Add", "Subtract", "Multiply"]
    )

    if st.button("Calculate"):
        if operation == "Add":
            result = a + b
        elif operation == "Subtract":
            result = a - b
        else:
            result = a * b

        st.write("Hello", name)
        st.write("Operation:", operation)
        st.write("Result:", result)

if __name__ == "__main__":
    main()
