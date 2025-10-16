import secrets
import string

def generate_secure_password(length=12):
    if length < 8:
        raise ValueError("Password length should be at least 8 characters for security reasons.")

    # Define character sets
    lowercase = string.ascii_lowercase
    uppercase = string.ascii_uppercase
    digits = string.digits
    punctuation = string.punctuation

    # Ensure the password contains at least one character from each category
    password_chars = [
        secrets.choice(lowercase),
        secrets.choice(uppercase),
        secrets.choice(digits),
        secrets.choice(punctuation)
    ]

    # Fill the rest of the password length with random choices from all categories
    all_characters = lowercase + uppercase + digits + punctuation
    password_chars += [secrets.choice(all_characters) for _ in range(length - 4)]

    # Shuffle the characters to avoid predictable patterns
    secrets.SystemRandom().shuffle(password_chars)

    # Join the list into a string to form the final password
    return ''.join(password_chars)

def main():
    # Generate a secure password
    password = generate_secure_password(12)
    print(f"Generated Secure Password: {password}")

if __name__ == "__main__":
    main()