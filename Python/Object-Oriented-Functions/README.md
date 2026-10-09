# Python & Object-Oriented Programming Fundamentals

Foundational Python exercises I worked through while building software-engineering skills for data science. They progress from procedural code to fully object-oriented designs with classes, encapsulation, multiple instances, and exception handling.

## Bank system — procedural → object-oriented

A simple banking application rebuilt step by step to show how and why OOP improves the design.

| Stage | Notebook(s) | Concept |
|---|---|---|
| Procedural | `Bank1_OneAccount` → `Bank5_Dictionary` | Global variables, then functions, then lists and dictionaries for multiple accounts |
| OOP v1 | `BankOOP1_IndividualVariables/` | First `Account` class with separate instance variables |
| OOP v2 | `BankOOP1_IndividualVariables/BankOOP2_ListOfAccountObjects/` | Managing a list of `Account` objects |
| OOP v3 | `BankOOP3_DictionaryOfAccount/` | Looking up accounts by number with a dictionary |
| OOP v4 | `BankOOP4_InteractiveMenu/` | Interactive command menu |
| OOP v5 | `BankOOP5_SeparateBankClass/` | A dedicated `Bank` class that manages the accounts |
| OOP v6 | `BankOOP6_UsingException/`, `BankOOP6_UsingExceptions/` | Custom exceptions for robust error handling |

`Account.ipynb` and `Bank_Class.ipynb` contain the standalone class definitions.

## Modeling objects and state

| Notebook | Concept |
|---|---|
| `LightSwitch`, `OO_LightSwitch_Two_Instances` | Simple class with on/off state; multiple independent instances |
| `DimmerSwitch`, `OO_DimmerSwitch_with_Test_Code` | State with bounded values plus test code |
| `TV`, `OO_TV_TwoInstances`, `OO_TV_with_Test_Code` | Richer object with several attributes and methods |
| `Time_Class` | Class modeling time values |
| `Monstor` | Simple game-character class |

## Algorithms, games, and tools

| Notebook | Description |
|---|---|
| `HigherOrLower` | Card game built procedurally with a deck of cards |
| `CountWhiteSpaces` | String processing exercise |
| `LongestCommonPrefix` | Classic string algorithm problem |
| `PygameDemo0_WindowOnly/` | Intro to the Pygame event loop |
| `Mitosheet` | Trying out the Mito spreadsheet interface for pandas |
