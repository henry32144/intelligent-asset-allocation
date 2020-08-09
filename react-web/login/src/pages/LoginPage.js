import React from 'react';
import Avatar from '@material-ui/core/Avatar';
import Button from '@material-ui/core/Button';
import CssBaseline from '@material-ui/core/CssBaseline';
import TextField from '@material-ui/core/TextField';
import FormControlLabel from '@material-ui/core/FormControlLabel';
import Checkbox from '@material-ui/core/Checkbox';
import Link from '@material-ui/core/Link';
import Grid from '@material-ui/core/Grid';
import Box from '@material-ui/core/Box';
import LockOutlinedIcon from '@material-ui/icons/LockOutlined';
import Typography from '@material-ui/core/Typography';
import { makeStyles } from '@material-ui/core/styles';
import Container from '@material-ui/core/Container';

import CircularProgress from '@material-ui/core/CircularProgress';
import Backdrop from '@material-ui/core/Backdrop';
import Cookies from 'universal-cookie';
import { BASEURL, SIGNUP_PAGE, DASHBOARD_PAGE } from '../Constants';

function Copyright() {
  return (
    <Typography variant="body2" color="textSecondary" align="center">
      {'Copyright © '}
      <Link color="inherit" href="https://127.0.0.1:5000/">
        HuggingMoney
      </Link>{' '}
      {new Date().getFullYear()}
      {'.'}
    </Typography>
  );
}

const useStyles = makeStyles((theme) => ({
  paper: {
    marginTop: theme.spacing(8),
    marginRight: theme.spacing(1),
    marginLeft: theme.spacing(1),
    display: 'flex',
    flexDirection: 'column',
    alignItems: 'center',
  },
  avatar: {
    margin: theme.spacing(1),
    backgroundColor: theme.palette.secondary.main,
  },
  form: {
    width: '100%', // Fix IE 11 issue.
    marginTop: theme.spacing(1),
  },
  inputBox: {
    margin: theme.spacing(0, 0, 2),
  },
  submit: {
    margin: theme.spacing(3, 0, 2),
  },
  imageIcon: {
    width: "60px",
    height: "60px",
  },
  errorMessage: {
    color: "#f44336",
    margin: theme.spacing(0, 2, 0),
  },
  backdrop: {
    zIndex: 1600,
    color: '#fff',
  },
}));

export default function LoginPage(props) {
  const classes = useStyles();

  const [emailInputIsError, setEmailError] = React.useState(false);
  const [passwordInputIsError, setPasswordError] = React.useState(false);
  const [backdropOpen, setBackdropOpen] = React.useState(false);

  const [emailInputErrorMsg, setEmailErrorMsg] = React.useState("");
  const [passwordInputMsg, setPasswordErrorMsg] = React.useState("");
  const [errorMsg, setErrorMsg] = React.useState("");
  const emailInput = React.useRef();
  const passwordInput = React.useRef();

  const handleBackdropClose = () => {
    setBackdropOpen(false);
  };
  const handleBackdropToggle = () => {
    setBackdropOpen(!backdropOpen);
  };

  const checkEmailInputEmpty = () => {
    if (emailInput.current.value.length < 1) {
      console.log("Email is empty")
      setEmailError(true);
      setEmailErrorMsg("Email cannot be empty");
    } else {
      setEmailError(false);
      setEmailErrorMsg("");
    }
  }

  const checkPasswordInputEmpty = () => {
    if (passwordInput.current.value.length < 1) {
      console.log("Password is empty")
      setPasswordError(true);
      setPasswordErrorMsg("Password cannot be empty");
    } else {
      setPasswordError(false);
      setPasswordErrorMsg("");
    }
  }

  // Do validate here when the focus of input field is out
  // Check whether the input is empty
  const emailInputOnBlur = () => {
    checkEmailInputEmpty();
  };

  const passwordInputOnBlur = () => {
    checkPasswordInputEmpty();
  };

  const signUpOnClick = async (e) => {
    window.location.pathname = './signup';
  }

  const loginButtonOnClick = async (e) => {
    // If no error in both textfield
    checkEmailInputEmpty();
    checkPasswordInputEmpty();
    setErrorMsg("");
    if (emailInput.current.value.length > 0 && passwordInput.current.value.length > 0) {
      // Do Login works
      // Assume authorization is pass
      const request = {
        method: 'POST',
        headers: {
          Accept: 'application/json',
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          'userEmail': emailInput.current.value,
          'userPassword': passwordInput.current.value,
        })
      }

      try {
        handleBackdropToggle();
        const response = await fetch(BASEURL + "/user/login", request)
        if (response.ok) {
          const jsonData = await response.json();
          if (jsonData.isSuccess) {
            // Temp user name
            const userId = jsonData.userId;
            const userName = jsonData.userName;
            const userEmail = emailInput.current.value;

            props.setUserData({
              userId: userId,
              userName: userName,
              userEmail: userEmail,
            });

            // Store user data into cookies
            const cookies = new Cookies();
            cookies.set('userId', userId, { path: '/' });
            cookies.set('userName', userName, { path: '/' });
            cookies.set('userEmail', userEmail, { path: '/' });
            window.location.pathname = './dashboard';
          } else {
            setErrorMsg(jsonData.errorMsg);
          }
        }
      }
      catch (err) {
        console.log('fetch failed', err);
      }
      finally {
        handleBackdropClose();
      }
    }
  };

  return (
    <div>
      <Backdrop className={classes.backdrop} open={backdropOpen} onClick={handleBackdropClose}>
        <CircularProgress color="inherit" />
      </Backdrop>
      <Container component="main" maxWidth="xs">
        <CssBaseline />
        <div className={classes.paper}>
          <img className={classes.imageIcon} src="../static/logo80.png" />
          <Typography component="h1" variant="h5">
            Log In
        </Typography>
          <div className={classes.form}>
            <TextField
              required
              name="email"
              autoComplete="email"
              autoFocus
              inputRef={emailInput}
              margin="normal"
              id="email"
              label="Email Address"
              type="email"
              variant="outlined"
              fullWidth
              error={emailInputIsError}
              helperText={emailInputErrorMsg}
              className={classes.inputBox}
              onBlur={emailInputOnBlur}
            />
            <TextField
              margin="normal"
              required
              name="password"
              autoComplete="current-password"
              inputRef={passwordInput}
              id="password"
              label="Password"
              type="password"
              variant="outlined"
              fullWidth
              error={passwordInputIsError}
              helperText={passwordInputMsg}
              className={classes.inputBox}
              onBlur={passwordInputOnBlur}
            />
            <Typography className={classes.errorMessage}>
              {errorMsg}
            </Typography>
            <Button
              type="submit"
              fullWidth
              variant="contained"
              color="primary"
              onClick={loginButtonOnClick}
              className={classes.submit}
            >
              Log In
            </Button>
            <Grid container>
              <Grid item>
                <Link href="./signup" variant="body2">
                  {"Don't have an account? Sign Up"}
                </Link>
              </Grid>
            </Grid>
          </div>
        </div>
        <Box mt={8}>
          <Copyright />
        </Box>
      </Container>
    </div>
  );
}