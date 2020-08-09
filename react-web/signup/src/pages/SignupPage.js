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
      <Link color="inherit" href="https://github.com/henry32144/intelligent-asset-allocation/">
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
    marginTop: theme.spacing(3),
  },
  imageIcon: {
    width: "60px",
    height: "60px",
  },
  inputBox: {
    margin: theme.spacing(0, 0, 2),
  },
  submit: {
    margin: theme.spacing(3, 0, 2),
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

export default function SignupPage(props) {
  const classes = useStyles();
  const [loading, setLoading] = React.useState(false);

  const [nameInputIsError, setNameError] = React.useState(false);
  const [emailInputIsError, setEmailError] = React.useState(false);
  const [passwordInputIsError, setPasswordError] = React.useState(false);
  const [backdropOpen, setBackdropOpen] = React.useState(false);

  const [nameInputErrorMsg, setNameErrorMsg] = React.useState("");
  const [emailInputErrorMsg, setEmailErrorMsg] = React.useState("");
  const [passwordInputMsg, setPasswordErrorMsg] = React.useState("");
  const [errorMsg, setErrorMsg] = React.useState("");

  const nameInput = React.useRef();
  const emailInput = React.useRef();
  const passwordInput = React.useRef();

  const handleBackdropClose = () => {
    setBackdropOpen(false);
  };
  const handleBackdropToggle = () => {
    setBackdropOpen(!backdropOpen);
  };

  const checkNameInputEmpty = () => {
    if (nameInput.current.value.length < 1) {
      console.log("Name is empty")
      setNameError(true);
      setNameErrorMsg("Name cannot be empty");
    } else {
      setNameError(false);
      setNameErrorMsg("");
    }
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
  };

  const checkPasswordInputEmpty = () => {
    if (passwordInput.current.value.length < 1) {
      console.log("Password is empty")
      setPasswordError(true);
      setPasswordErrorMsg("Password cannot be empty");
    } else {
      setPasswordError(false);
      setPasswordErrorMsg("");
    }
  };

  // Do validate here when the focus of input field is out
  // Check whether the input is empty
  const nameInputOnBlur = () => {
    checkNameInputEmpty();
  };

  const emailInputOnBlur = () => {
    checkEmailInputEmpty();
  };

  const passwordInputOnBlur = () => {
    checkPasswordInputEmpty();
  };

  const signUpSucceed = () => {
    props.setDialogMessage("Sign up success! Will be redirect in 3 seconds");
    props.openMessageDialog();
    window.setTimeout(()=>{window.location.pathname = './';}, 3000);
  };

  const signUpButtonOnClick = async (e) => {
    checkNameInputEmpty();
    checkEmailInputEmpty();
    checkPasswordInputEmpty();
    // If no error in both textfield
    if (nameInput.current.value.length > 0 &&
      emailInput.current.value.length > 0 &&
      passwordInput.current.value.length > 0) {

      const request = {
        method: 'POST',
        headers: {
          Accept: 'application/json',
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          'userName': nameInput.current.value,
          'userEmail': emailInput.current.value,
          'userPassword': passwordInput.current.value,
        })
      }

      try {
        handleBackdropToggle(true);
        const response = await fetch(BASEURL + "/user/signup", request)
        if (response.ok) {
          const jsonData = await response.json();
          if (jsonData.isSuccess) {
            signUpSucceed();
          } else {
            setErrorMsg(jsonData.errorMsg);
          }
        }
      }
      catch (err) {
        alert('fetch failed', err);
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
            Sign up
        </Typography>
          <div className={classes.form}>
            <TextField
              required
              name="name"
              autoFocus
              inputRef={nameInput}
              margin="normal"
              id="name"
              label="User Name"
              type="text"
              variant="outlined"
              fullWidth
              error={nameInputIsError}
              helperText={nameInputErrorMsg}
              className={classes.inputBox}
              onBlur={nameInputOnBlur}
            />
            <TextField
              required
              name="email"
              autoComplete="email"
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
              required
              name="password"
              autoComplete="current-password"
              inputRef={passwordInput}
              margin="normal"
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
              variant="contained" 
              onClick={signUpButtonOnClick} 
              color="primary" 
              fullWidth
              className={classes.submit}
            >
              Sign up
            </Button>
            <Grid container justify="flex-end">
              <Grid item>
                <Link href="/login" variant="body2">
                  Already have an account? Sign in
              </Link>
              </Grid>
            </Grid>
          </div>
        </div>
        <Box mt={5}>
          <Copyright />
        </Box>
      </Container>
    </div>
  );
}